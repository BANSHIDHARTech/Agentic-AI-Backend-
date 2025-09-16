-- Add unique constraint for workflow_steps table
-- This is needed for the ON CONFLICT clause in log_workflow_step function

-- First check if the constraint already exists to avoid errors
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_constraint 
        WHERE conname = 'workflow_steps_run_node_key'
    ) THEN
        -- Add unique constraint
        ALTER TABLE workflow_steps 
        ADD CONSTRAINT workflow_steps_run_node_key 
        UNIQUE (workflow_run_id, node_id);
    END IF;
END $$;

-- Also update the log_workflow_step function to handle the potential error more gracefully
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
    v_step_exists BOOLEAN;
BEGIN
    -- Check if the step already exists
    SELECT EXISTS(
        SELECT 1 
        FROM workflow_steps 
        WHERE workflow_run_id = p_workflow_run_id 
        AND node_id = p_node_id
    ) INTO v_step_exists;
    
    IF v_step_exists THEN
        -- Update existing step
        UPDATE workflow_steps
        SET status = p_status,
            execution_context = 
                CASE 
                    WHEN p_execution_context IS NOT NULL AND p_execution_context != '{}'::jsonb
                    THEN workflow_steps.execution_context || p_execution_context
                    ELSE workflow_steps.execution_context
                END,
            error_message = COALESCE(p_error_message, workflow_steps.error_message),
            updated_at = NOW()
        WHERE workflow_run_id = p_workflow_run_id 
        AND node_id = p_node_id
        RETURNING id INTO v_step_id;
    ELSE
        -- Insert new step
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
        RETURNING id INTO v_step_id;
    END IF;
    
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