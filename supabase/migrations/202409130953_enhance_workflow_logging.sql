-- Enhanced workflow logging migration
-- This migration adds execution context and status tracking to workflow steps

-- Add new columns to workflow_steps if they don't exist
DO $$
BEGIN
    -- Add status column if not exists
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_steps' AND column_name = 'status'
    ) THEN
        ALTER TABLE workflow_steps 
        ADD COLUMN status TEXT 
        DEFAULT 'pending' 
        CHECK (status IN ('pending', 'running', 'completed', 'failed'))
        NOT NULL;
    END IF;

    -- Add execution_context column if not exists
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_steps' AND column_name = 'execution_context'
    ) THEN
        ALTER TABLE workflow_steps 
        ADD COLUMN execution_context JSONB 
        DEFAULT '{}'::jsonb;
    END IF;

    -- Add timing columns if they don't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_steps' AND column_name = 'started_at'
    ) THEN
        ALTER TABLE workflow_steps 
        ADD COLUMN started_at TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'workflow_steps' AND column_name = 'ended_at'
    ) THEN
        ALTER TABLE workflow_steps 
        ADD COLUMN ended_at TIMESTAMPTZ;
    END IF;

    -- Add index for status if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_indexes 
        WHERE indexname = 'idx_workflow_steps_status'
    ) THEN
        CREATE INDEX idx_workflow_steps_status ON workflow_steps(status);
    END IF;

    -- Add index for workflow_run_id if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_indexes 
        WHERE indexname = 'idx_workflow_steps_run_id'
    ) THEN
        CREATE INDEX idx_workflow_steps_run_id ON workflow_steps(workflow_run_id);
    END IF;
END $$;

-- Create or replace function to update timestamps
CREATE OR REPLACE FUNCTION update_workflow_step_timestamps()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'running' AND OLD.status = 'pending' THEN
        NEW.started_at = COALESCE(NEW.started_at, NOW());
    ELSIF NEW.status IN ('completed', 'failed') AND OLD.status = 'running' THEN
        NEW.ended_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updating timestamps
DROP TRIGGER IF EXISTS trigger_update_workflow_step_timestamps ON workflow_steps;
CREATE TRIGGER trigger_update_workflow_step_timestamps
BEFORE UPDATE ON workflow_steps
FOR EACH ROW
WHEN (
    NEW.status IS DISTINCT FROM OLD.status
)
EXECUTE FUNCTION update_workflow_step_timestamps();

-- Create or replace function to log workflow step execution
CREATE OR REPLACE FUNCTION log_workflow_step(
    p_workflow_run_id UUID,
    p_node_id UUID,
    p_status TEXT,
    p_execution_context JSONB DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_step_id UUID;
BEGIN
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
        NOW(),
        NOW()
    )
    ON CONFLICT (workflow_run_id, node_id) 
    DO UPDATE SET
        status = EXCLUDED.status,
        execution_context = 
            CASE 
                WHEN EXCLUDED.execution_context IS NOT NULL AND EXCLUDED.execution_context != '{}'::jsonb
                THEN workflow_steps.execution_context || EXCLUDED.execution_context
                ELSE workflow_steps.execution_context
            END,
        error_message = COALESCE(EXCLUDED.error_message, workflow_steps.error_message),
        updated_at = NOW()
    RETURNING id INTO v_step_id;
    
    -- Update timestamps based on status
    IF p_status = 'running' THEN
        UPDATE workflow_steps
        SET started_at = NOW()
        WHERE id = v_step_id AND started_at IS NULL;
    ELSIF p_status IN ('completed', 'failed') THEN
        UPDATE workflow_steps
        SET ended_at = NOW()
        WHERE id = v_step_id AND ended_at IS NULL;
    END IF;
    
    RETURN v_step_id;
END;
$$ LANGUAGE plpgsql;
