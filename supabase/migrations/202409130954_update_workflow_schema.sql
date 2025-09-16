-- Migration to update workflow schema with consistent column names
-- and add missing tables/functions

-- 1. Add workflow_steps table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.workflow_steps (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_run_id uuid NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    node_id uuid NOT NULL REFERENCES workflow_nodes(id) ON DELETE CASCADE,
    status text DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    input_data jsonb DEFAULT '{}'::jsonb,
    output_data jsonb,
    error_message text,
    execution_context jsonb DEFAULT '{}'::jsonb,
    started_at timestamptz,
    ended_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- 2. Create indexes for workflow_steps
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_run_id ON workflow_steps(workflow_run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_node_id ON workflow_steps(node_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON workflow_steps(status);

-- 3. Enable RLS on workflow_steps
ALTER TABLE workflow_steps ENABLE ROW LEVEL SECURITY;

-- 4. Add RLS policies for workflow_steps
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'workflow_steps' AND policyname = 'Enable read access for all users') THEN
        CREATE POLICY "Enable read access for all users" ON workflow_steps FOR SELECT USING (true);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'workflow_steps' AND policyname = 'Enable insert for all users') THEN
        CREATE POLICY "Enable insert for all users" ON workflow_steps FOR INSERT WITH CHECK (true);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'workflow_steps' AND policyname = 'Enable update for all users') THEN
        CREATE POLICY "Enable update for all users" ON workflow_steps FOR UPDATE USING (true);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'workflow_steps' AND policyname = 'Enable delete for all users') THEN
        CREATE POLICY "Enable delete for all users" ON workflow_steps FOR DELETE USING (true);
    END IF;
END $$;

-- 5. Create or replace the update_workflow_step_timestamps function
CREATE OR REPLACE FUNCTION update_workflow_step_timestamps()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'running' AND OLD.status = 'pending' AND NEW.started_at IS NULL THEN
        NEW.started_at = now();
    ELSIF NEW.status IN ('completed', 'failed') AND OLD.status = 'running' AND NEW.ended_at IS NULL THEN
        NEW.ended_at = now();
    END IF;
    
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 6. Create trigger for workflow_steps timestamps
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_trigger 
        WHERE tgname = 'update_workflow_step_timestamps_trigger'
    ) THEN
        CREATE TRIGGER update_workflow_step_timestamps_trigger
        BEFORE UPDATE ON workflow_steps
        FOR EACH ROW
        EXECUTE FUNCTION update_workflow_step_timestamps();
    END IF;
END $$;

-- 7. Add any missing columns to workflow_edges
DO $$
BEGIN
    -- Add source_handle if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_edges' AND column_name = 'source_handle'
    ) THEN
        ALTER TABLE workflow_edges ADD COLUMN source_handle TEXT NOT NULL DEFAULT 'output';
    END IF;
    
    -- Add target_handle if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_edges' AND column_name = 'target_handle'
    ) THEN
        ALTER TABLE workflow_edges ADD COLUMN target_handle TEXT NOT NULL DEFAULT 'input';
    END IF;
    
    -- Add data column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_edges' AND column_name = 'data'
    ) THEN
        ALTER TABLE workflow_edges ADD COLUMN data JSONB NOT NULL DEFAULT '{}'::jsonb;
    END IF;
    
    -- Add updated_at column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_edges' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE workflow_edges ADD COLUMN updated_at TIMESTAMPTZ DEFAULT now();
    END IF;
END $$;

-- 8. Create indexes for workflow_edges if they don't exist
DO $$
BEGIN
    -- Index on workflow_id
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_indexes 
        WHERE indexname = 'idx_workflow_edges_workflow_id'
    ) THEN
        CREATE INDEX idx_workflow_edges_workflow_id ON workflow_edges(workflow_id);
    END IF;
    
    -- Index on from_node_id
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_indexes 
        WHERE indexname = 'idx_workflow_edges_from_node'
    ) THEN
        CREATE INDEX idx_workflow_edges_from_node ON workflow_edges(from_node_id);
    END IF;
    
    -- Index on to_node_id
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_indexes 
        WHERE indexname = 'idx_workflow_edges_to_node'
    ) THEN
        CREATE INDEX idx_workflow_edges_to_node ON workflow_edges(to_node_id);
    END IF;
END $$;

-- 9. Create or replace the log_workflow_step function
CREATE OR REPLACE FUNCTION log_workflow_step(
    p_workflow_run_id uuid,
    p_node_id uuid,
    p_status text,
    p_execution_context jsonb DEFAULT NULL,
    p_error_message text DEFAULT NULL
)
RETURNS uuid AS $$
DECLARE
    v_step_id uuid;
    v_workflow_id uuid;
BEGIN
    -- Get workflow_id for logging
    SELECT workflow_id INTO v_workflow_id
    FROM workflow_runs
    WHERE id = p_workflow_run_id;
    
    -- Insert or update the step
    INSERT INTO workflow_steps (
        workflow_run_id,
        node_id,
        status,
        execution_context,
        error_message,
        created_at,
        updated_at
    )
    VALUES (
        p_workflow_run_id,
        p_node_id,
        p_status,
        COALESCE(p_execution_context, '{}'::jsonb),
        p_error_message,
        now(),
        now()
    )
    ON CONFLICT (workflow_run_id, node_id) 
    DO UPDATE SET
        status = EXCLUDED.status,
        execution_context = 
            CASE 
                WHEN EXCLUDED.status = 'running' AND workflow_steps.status != 'running' 
                THEN EXCLUDED.execution_context
                ELSE workflow_steps.execution_context || EXCLUDED.execution_context
            END,
        error_message = COALESCE(EXCLUDED.error_message, workflow_steps.error_message),
        updated_at = now()
    RETURNING id INTO v_step_id;
    
    -- Log the step update
    INSERT INTO logs (
        event_type,
        workflow_id,
        workflow_run_id,
        node_id,
        details
    )
    VALUES (
        'workflow_step_' || p_status,
        v_workflow_id,
        p_workflow_run_id,
        p_node_id,
        jsonb_build_object(
            'step_id', v_step_id,
            'status', p_status,
            'timestamp', now()
        )
    );
    
    RETURN v_step_id;
END;
$$ LANGUAGE plpgsql;
