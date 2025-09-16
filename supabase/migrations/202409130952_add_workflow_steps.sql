-- Create workflow_steps table for tracking individual workflow step execution
CREATE TABLE IF NOT EXISTS workflow_steps (
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

-- Create indexes for workflow_steps
CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow_run_id ON workflow_steps(workflow_run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_node_id ON workflow_steps(node_id);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON workflow_steps(status);

-- Enable RLS
ALTER TABLE workflow_steps ENABLE ROW LEVEL SECURITY;

-- Add RLS policies
CREATE POLICY "Enable read access for all users" ON workflow_steps FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON workflow_steps FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON workflow_steps FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON workflow_steps FOR DELETE USING (true);

-- Add comments
COMMENT ON TABLE workflow_steps IS 'Tracks execution of individual steps within a workflow run';
COMMENT ON COLUMN workflow_steps.status IS 'Current status of the workflow step';
COMMENT ON COLUMN workflow_steps.execution_context IS 'Contextual data for the step execution';

-- Add function to update timestamps
CREATE OR REPLACE FUNCTION update_workflow_step_timestamps()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for timestamps
CREATE TRIGGER update_workflow_steps_updated_at
BEFORE UPDATE ON workflow_steps
FOR EACH ROW
EXECUTE FUNCTION update_workflow_step_timestamps();
