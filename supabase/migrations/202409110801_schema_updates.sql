-- Add missing columns with IF NOT EXISTS to be idempotent

-- Add session_id to workflows
ALTER TABLE workflows 
ADD COLUMN IF NOT EXISTS session_id uuid;

-- Add data and tool_id to workflow_nodes
ALTER TABLE workflow_nodes 
ADD COLUMN IF NOT EXISTS data jsonb,
ADD COLUMN IF NOT EXISTS tool_id uuid REFERENCES tools(id) ON DELETE SET NULL;

-- Add missing columns to logs
ALTER TABLE logs 
ADD COLUMN IF NOT EXISTS node_id uuid,
ADD COLUMN IF NOT EXISTS session_id uuid,
ADD COLUMN IF NOT EXISTS workflow_id uuid REFERENCES workflows(id) ON DELETE CASCADE,
ADD COLUMN IF NOT EXISTS output jsonb;

-- Add session_id to workflow_runs
ALTER TABLE workflow_runs 
ADD COLUMN IF NOT EXISTS session_id uuid;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_workflow_nodes_tool_id ON workflow_nodes(tool_id);
CREATE INDEX IF NOT EXISTS idx_logs_workflow_id ON logs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_logs_node_id ON logs(node_id);
CREATE INDEX IF NOT EXISTS idx_logs_session_id ON logs(session_id);
CREATE INDEX IF NOT EXISTS idx_workflows_session_id ON workflows(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_session_id ON workflow_runs(session_id);

-- Update RLS policies for new columns
ALTER TABLE workflow_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE logs ENABLE ROW LEVEL SECURITY;

-- Add comments for documentation
COMMENT ON COLUMN workflows.session_id IS 'Session ID for workflow execution context';
COMMENT ON COLUMN workflow_nodes.data IS 'Additional node configuration data';
COMMENT ON COLUMN workflow_nodes.tool_id IS 'Reference to tools table for tool nodes';
COMMENT ON COLUMN logs.node_id IS 'Node that generated this log entry';
COMMENT ON COLUMN logs.session_id IS 'Session ID for grouping related logs';
COMMENT ON COLUMN logs.workflow_id IS 'Workflow that generated this log entry';
COMMENT ON COLUMN logs.output IS 'Structured output data from the node';
COMMENT ON COLUMN workflow_runs.session_id IS 'Session ID for workflow run context';
