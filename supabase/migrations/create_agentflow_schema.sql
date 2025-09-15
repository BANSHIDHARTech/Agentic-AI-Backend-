/*
  # AgentFlow Complete Database Schema - Idempotent Version
  
  This migration creates all required tables for the AgentFlow AI agent orchestration platform.
  It's designed to be completely idempotent and safe to run multiple times.
  
  1. Core Tables
    - `tools` - Callable functions that agents can execute
    - `agents` - AI agents with prompts, intents, and optional tool bindings
    - `workflows` - Workflow definitions and metadata
    - `workflow_nodes` - Agents within workflows (DAG nodes)
    - `workflow_edges` - Intent-based transitions between workflow nodes
    - `workflow_runs` - Workflow execution instances and state tracking
    - `logs` - Comprehensive execution and event logging

  2. Security
    - Row Level Security (RLS) enabled on all tables
    - Idempotent policies that can be safely re-run
    - Proper foreign key constraints and cascading deletes

  3. Performance
    - Comprehensive indexes for all common query patterns
    - GIN indexes for JSONB columns
    - Unique constraints where appropriate
*/

-- Drop existing policies if they exist (idempotent)
DROP POLICY IF EXISTS "Enable read access for all users" ON tools;
DROP POLICY IF EXISTS "Enable insert for all users" ON tools;
DROP POLICY IF EXISTS "Enable update for all users" ON tools;
DROP POLICY IF EXISTS "Enable delete for all users" ON tools;

DROP POLICY IF EXISTS "Enable read access for all users" ON agents;
DROP POLICY IF EXISTS "Enable insert for all users" ON agents;
DROP POLICY IF EXISTS "Enable update for all users" ON agents;
DROP POLICY IF EXISTS "Enable delete for all users" ON agents;

DROP POLICY IF EXISTS "Enable read access for all users" ON workflows;
DROP POLICY IF EXISTS "Enable insert for all users" ON workflows;
DROP POLICY IF EXISTS "Enable update for all users" ON workflows;
DROP POLICY IF EXISTS "Enable delete for all users" ON workflows;

DROP POLICY IF EXISTS "Enable read access for all users" ON workflow_nodes;
DROP POLICY IF EXISTS "Enable insert for all users" ON workflow_nodes;
DROP POLICY IF EXISTS "Enable update for all users" ON workflow_nodes;
DROP POLICY IF EXISTS "Enable delete for all users" ON workflow_nodes;

DROP POLICY IF EXISTS "Enable read access for all users" ON workflow_edges;
DROP POLICY IF EXISTS "Enable insert for all users" ON workflow_edges;
DROP POLICY IF EXISTS "Enable update for all users" ON workflow_edges;
DROP POLICY IF EXISTS "Enable delete for all users" ON workflow_edges;

DROP POLICY IF EXISTS "Enable read access for all users" ON workflow_runs;
DROP POLICY IF EXISTS "Enable insert for all users" ON workflow_runs;
DROP POLICY IF EXISTS "Enable update for all users" ON workflow_runs;
DROP POLICY IF EXISTS "Enable delete for all users" ON workflow_runs;

DROP POLICY IF EXISTS "Enable read access for all users" ON logs;
DROP POLICY IF EXISTS "Enable insert for all users" ON logs;
DROP POLICY IF EXISTS "Enable update for all users" ON logs;
DROP POLICY IF EXISTS "Enable delete for all users" ON logs;

-- Drop existing indexes if they exist (idempotent)
DROP INDEX IF EXISTS idx_agents_name;
DROP INDEX IF EXISTS idx_agents_input_intents;
DROP INDEX IF EXISTS idx_agents_output_intents;
DROP INDEX IF EXISTS idx_agents_is_active;
DROP INDEX IF EXISTS idx_agents_tool_id;

DROP INDEX IF EXISTS idx_tools_name;
DROP INDEX IF EXISTS idx_tools_is_active;

DROP INDEX IF EXISTS idx_workflows_name;
DROP INDEX IF EXISTS idx_workflows_is_active;

DROP INDEX IF EXISTS idx_workflow_nodes_workflow_id;
DROP INDEX IF EXISTS idx_workflow_nodes_agent_id;
DROP INDEX IF EXISTS idx_workflow_nodes_node_type;

DROP INDEX IF EXISTS idx_workflow_edges_workflow_id;
DROP INDEX IF EXISTS idx_workflow_edges_from_node;
DROP INDEX IF EXISTS idx_workflow_edges_to_node;
DROP INDEX IF EXISTS idx_workflow_edges_trigger_intent;

DROP INDEX IF EXISTS idx_workflow_runs_workflow_id;
DROP INDEX IF EXISTS idx_workflow_runs_status;
DROP INDEX IF EXISTS idx_workflow_runs_started_at;
DROP INDEX IF EXISTS idx_workflow_runs_completed_at;

DROP INDEX IF EXISTS idx_logs_event_type;
DROP INDEX IF EXISTS idx_logs_workflow_run_id;
DROP INDEX IF EXISTS idx_logs_created_at;

-- Drop existing views if they exist (idempotent)
DROP VIEW IF EXISTS workflow_summary;
DROP VIEW IF EXISTS agent_tool_summary;
DROP VIEW IF EXISTS recent_workflow_runs;
DROP VIEW IF EXISTS workflow_performance;
DROP VIEW IF EXISTS agent_usage_stats;

-- Drop existing functions if they exist (idempotent)
DROP FUNCTION IF EXISTS get_workflow_stats(uuid);

-- Create tools table first (referenced by agents)
CREATE TABLE IF NOT EXISTS tools (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text UNIQUE NOT NULL,
  description text DEFAULT '',
  function_code text NOT NULL,
  parameters jsonb DEFAULT '{}'::jsonb,
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text UNIQUE NOT NULL,
  description text DEFAULT '',
  system_prompt text NOT NULL,
  model_name text NOT NULL DEFAULT 'gpt-4',
  input_intents jsonb DEFAULT '[]'::jsonb,
  output_intents jsonb DEFAULT '[]'::jsonb,
  tool_id uuid REFERENCES tools(id) ON DELETE SET NULL,
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create workflows table
CREATE TABLE IF NOT EXISTS workflows (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text UNIQUE NOT NULL,
  description text DEFAULT '',
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create workflow_nodes table (agents within workflows)
CREATE TABLE IF NOT EXISTS workflow_nodes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id uuid NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  agent_id uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  node_type text DEFAULT 'agent' CHECK (node_type IN ('start', 'agent', 'end')),
  position jsonb DEFAULT '{"x": 0, "y": 0}'::jsonb,
  created_at timestamptz DEFAULT now(),
  UNIQUE(workflow_id, agent_id)
);

-- Create workflow_edges table (intent-based transitions)
CREATE TABLE IF NOT EXISTS workflow_edges (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id uuid NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  from_node_id uuid NOT NULL REFERENCES workflow_nodes(id) ON DELETE CASCADE,
  to_node_id uuid NOT NULL REFERENCES workflow_nodes(id) ON DELETE CASCADE,
  trigger_intent text NOT NULL,
  condition jsonb DEFAULT null,
  created_at timestamptz DEFAULT now(),
  UNIQUE(workflow_id, from_node_id, trigger_intent)
);

-- Create workflow_runs table (execution tracking)
CREATE TABLE IF NOT EXISTS workflow_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id uuid NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  status text DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
  input_data jsonb DEFAULT '{}'::jsonb,
  output_data jsonb DEFAULT '{}'::jsonb,
  execution_path jsonb DEFAULT '[]'::jsonb,
  error_message text,
  started_at timestamptz DEFAULT now(),
  completed_at timestamptz,
  created_at timestamptz DEFAULT now()
);

-- Create logs table (comprehensive event logging)
CREATE TABLE IF NOT EXISTS logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type text NOT NULL,
  workflow_run_id uuid REFERENCES workflow_runs(id) ON DELETE CASCADE,
  details jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);

-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
CREATE INDEX IF NOT EXISTS idx_agents_input_intents ON agents USING gin(input_intents);
CREATE INDEX IF NOT EXISTS idx_agents_output_intents ON agents USING gin(output_intents);
CREATE INDEX IF NOT EXISTS idx_agents_is_active ON agents(is_active);
CREATE INDEX IF NOT EXISTS idx_agents_tool_id ON agents(tool_id);

CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);
CREATE INDEX IF NOT EXISTS idx_tools_is_active ON tools(is_active);

CREATE INDEX IF NOT EXISTS idx_workflows_name ON workflows(name);
CREATE INDEX IF NOT EXISTS idx_workflows_is_active ON workflows(is_active);

CREATE INDEX IF NOT EXISTS idx_workflow_nodes_workflow_id ON workflow_nodes(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_nodes_agent_id ON workflow_nodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_workflow_nodes_node_type ON workflow_nodes(node_type);

CREATE INDEX IF NOT EXISTS idx_workflow_edges_workflow_id ON workflow_edges(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_edges_from_node ON workflow_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_workflow_edges_to_node ON workflow_edges(to_node_id);
CREATE INDEX IF NOT EXISTS idx_workflow_edges_trigger_intent ON workflow_edges(trigger_intent);

CREATE INDEX IF NOT EXISTS idx_workflow_runs_workflow_id ON workflow_runs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_started_at ON workflow_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_completed_at ON workflow_runs(completed_at);

CREATE INDEX IF NOT EXISTS idx_logs_event_type ON logs(event_type);
CREATE INDEX IF NOT EXISTS idx_logs_workflow_run_id ON logs(workflow_run_id);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at);

-- Enable Row Level Security (RLS)
ALTER TABLE tools ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE logs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (permissive for development - customize for production)
-- Tools policies
CREATE POLICY "Enable read access for all users" ON tools FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON tools FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON tools FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON tools FOR DELETE USING (true);

-- Agents policies
CREATE POLICY "Enable read access for all users" ON agents FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON agents FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON agents FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON agents FOR DELETE USING (true);

-- Workflows policies
CREATE POLICY "Enable read access for all users" ON workflows FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON workflows FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON workflows FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON workflows FOR DELETE USING (true);

-- Workflow nodes policies
CREATE POLICY "Enable read access for all users" ON workflow_nodes FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON workflow_nodes FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON workflow_nodes FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON workflow_nodes FOR DELETE USING (true);

-- Workflow edges policies
CREATE POLICY "Enable read access for all users" ON workflow_edges FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON workflow_edges FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON workflow_edges FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON workflow_edges FOR DELETE USING (true);

-- Workflow runs policies
CREATE POLICY "Enable read access for all users" ON workflow_runs FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON workflow_runs FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON workflow_runs FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON workflow_runs FOR DELETE USING (true);

-- Logs policies
CREATE POLICY "Enable read access for all users" ON logs FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON logs FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON logs FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON logs FOR DELETE USING (true);

-- Add helpful comments for documentation
COMMENT ON TABLE tools IS 'Callable functions that agents can execute to perform specific tasks';
COMMENT ON TABLE agents IS 'AI agents with system prompts, intents, and optional tool bindings';
COMMENT ON TABLE workflows IS 'Workflow definitions and metadata for agent orchestration';
COMMENT ON TABLE workflow_nodes IS 'Agents within workflows representing DAG nodes';
COMMENT ON TABLE workflow_edges IS 'Intent-based transitions between workflow nodes';
COMMENT ON TABLE workflow_runs IS 'Workflow execution instances with state tracking';
COMMENT ON TABLE logs IS 'Comprehensive execution and event logging for debugging and monitoring';

-- Column comments for better understanding
COMMENT ON COLUMN agents.input_intents IS 'JSON array of intents this agent can handle (e.g., ["need_authentication"])';
COMMENT ON COLUMN agents.output_intents IS 'JSON array of intents this agent can produce (e.g., ["need_balance_info"])';
COMMENT ON COLUMN agents.system_prompt IS 'LLM system prompt that defines the agent behavior and personality';
COMMENT ON COLUMN agents.tool_id IS 'Optional reference to a tool this agent can execute';

COMMENT ON COLUMN workflow_edges.trigger_intent IS 'Intent that triggers this transition between nodes';
COMMENT ON COLUMN workflow_edges.condition IS 'Optional JSON condition for conditional routing';

COMMENT ON COLUMN workflow_runs.execution_path IS 'JSON array containing step-by-step execution history';
COMMENT ON COLUMN workflow_runs.input_data IS 'Initial input data provided to start the workflow';
COMMENT ON COLUMN workflow_runs.output_data IS 'Final output data produced by the workflow';

COMMENT ON COLUMN logs.event_type IS 'Type of event (e.g., workflow_start, agent_execution, tool_call)';
COMMENT ON COLUMN logs.details IS 'JSON object containing event-specific data and context';

-- Create helpful views for easier querying and monitoring
CREATE OR REPLACE VIEW workflow_summary AS
SELECT 
  w.id,
  w.name,
  w.description,
  COUNT(DISTINCT wn.id) as node_count,
  COUNT(DISTINCT we.id) as edge_count,
  w.is_active,
  w.created_at,
  w.updated_at
FROM workflows w
LEFT JOIN workflow_nodes wn ON w.id = wn.workflow_id
LEFT JOIN workflow_edges we ON w.id = we.workflow_id
GROUP BY w.id, w.name, w.description, w.is_active, w.created_at, w.updated_at
ORDER BY w.created_at DESC;

CREATE OR REPLACE VIEW agent_tool_summary AS
SELECT 
  a.id,
  a.name as agent_name,
  a.description as agent_description,
  a.input_intents,
  a.output_intents,
  t.name as tool_name,
  t.description as tool_description,
  a.is_active,
  a.created_at
FROM agents a
LEFT JOIN tools t ON a.tool_id = t.id
ORDER BY a.name;

CREATE OR REPLACE VIEW recent_workflow_runs AS
SELECT 
  wr.id,
  w.name as workflow_name,
  wr.status,
  wr.started_at,
  wr.completed_at,
  EXTRACT(EPOCH FROM (COALESCE(wr.completed_at, now()) - wr.started_at)) as duration_seconds,
  jsonb_array_length(COALESCE(wr.execution_path, '[]'::jsonb)) as steps_executed
FROM workflow_runs wr
JOIN workflows w ON wr.workflow_id = w.id
ORDER BY wr.started_at DESC
LIMIT 50;

CREATE OR REPLACE VIEW workflow_performance AS
SELECT 
  w.name as workflow_name,
  COUNT(wr.id) as total_executions,
  COUNT(wr.id) FILTER (WHERE wr.status = 'completed') as successful_executions,
  COUNT(wr.id) FILTER (WHERE wr.status = 'failed') as failed_executions,
  ROUND(AVG(EXTRACT(EPOCH FROM (wr.completed_at - wr.started_at))), 2) as avg_duration_seconds,
  MAX(wr.started_at) as last_execution
FROM workflows w
LEFT JOIN workflow_runs wr ON w.id = wr.workflow_id
WHERE w.is_active = true
GROUP BY w.id, w.name
ORDER BY total_executions DESC;

CREATE OR REPLACE VIEW agent_usage_stats AS
SELECT 
  a.name as agent_name,
  COUNT(wn.id) as workflows_used_in,
  a.input_intents,
  a.output_intents,
  t.name as tool_name,
  a.is_active
FROM agents a
LEFT JOIN workflow_nodes wn ON a.id = wn.agent_id
LEFT JOIN tools t ON a.tool_id = t.id
WHERE a.is_active = true
GROUP BY a.id, a.name, a.input_intents, a.output_intents, t.name, a.is_active
ORDER BY workflows_used_in DESC, a.name;

-- Add comments for the views
COMMENT ON VIEW workflow_summary IS 'Summary view of all workflows with node and edge counts';
COMMENT ON VIEW agent_tool_summary IS 'Summary view of all agents with their associated tools';
COMMENT ON VIEW recent_workflow_runs IS 'Recent workflow executions with performance metrics';
COMMENT ON VIEW workflow_performance IS 'Performance metrics and statistics for all workflows';
COMMENT ON VIEW agent_usage_stats IS 'Usage statistics and tool associations for all agents';

-- Create a function to get workflow execution statistics
CREATE OR REPLACE FUNCTION get_workflow_stats(workflow_uuid uuid)
RETURNS TABLE(
  total_runs bigint,
  successful_runs bigint,
  failed_runs bigint,
  avg_duration_seconds numeric,
  last_run_at timestamptz
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE status = 'completed') as successful_runs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds,
    MAX(started_at) as last_run_at
  FROM workflow_runs 
  WHERE workflow_id = workflow_uuid;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_workflow_stats IS 'Get execution statistics for a specific workflow';

-- Final success message
DO $$
BEGIN
  RAISE NOTICE '=== AgentFlow Database Schema Created Successfully ===';
  RAISE NOTICE 'Tables: tools, agents, workflows, workflow_nodes, workflow_edges, workflow_runs, logs';
  RAISE NOTICE 'Views: workflow_summary, agent_tool_summary, recent_workflow_runs, workflow_performance, agent_usage_stats';
  RAISE NOTICE 'Functions: get_workflow_stats';
  RAISE NOTICE 'Ready for seed data insertion.';
  RAISE NOTICE '======================================================';
END $$;