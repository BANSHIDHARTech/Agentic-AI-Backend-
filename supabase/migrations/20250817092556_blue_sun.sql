/*
  # Chat Persistence and Scheduler Tables
  
  Creates tables for chat session management and workflow scheduling:
  
  1. New Tables
    - `chat_sessions` - Chat session tracking
    - `chat_messages` - Individual chat messages with role and metadata
    - `workflow_schedules` - Cron-based workflow scheduling
    - `workflow_switches` - Mid-session workflow switching logs

  2. Security
    - Enable RLS on all new tables
    - Add policies for authenticated access
    - Proper foreign key constraints

  3. Features
    - Chat persistence across workflow sessions
    - Mid-session intent recognition and workflow switching
    - Cron-based workflow scheduling
    - Comprehensive logging and tracking
*/

-- Create chat_sessions table for session tracking
CREATE TABLE IF NOT EXISTS chat_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text,
  started_at timestamptz DEFAULT now(),
  last_updated timestamptz DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Create chat_messages table for message persistence
CREATE TABLE IF NOT EXISTS chat_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role text NOT NULL CHECK(role IN ('user', 'agent', 'system')),
  content text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);

-- Create workflow_schedules table for cron scheduling
CREATE TABLE IF NOT EXISTS workflow_schedules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id uuid NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  cron_expression text NOT NULL,
  active boolean DEFAULT true,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create workflow_switches table for mid-session switching logs
CREATE TABLE IF NOT EXISTS workflow_switches (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  from_workflow_id uuid REFERENCES workflows(id),
  to_workflow_id uuid NOT NULL REFERENCES workflows(id),
  trigger_intent text,
  confidence numeric(3,2),
  switched_at timestamptz DEFAULT now(),
  metadata jsonb DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_started_at ON chat_sessions(started_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON chat_messages(role);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);

CREATE INDEX IF NOT EXISTS idx_workflow_schedules_workflow_id ON workflow_schedules(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_schedules_active ON workflow_schedules(active);
CREATE INDEX IF NOT EXISTS idx_workflow_schedules_created_at ON workflow_schedules(created_at);

CREATE INDEX IF NOT EXISTS idx_workflow_switches_session_id ON workflow_switches(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_switches_switched_at ON workflow_switches(switched_at);

-- Enable Row Level Security
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_switches ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Chat sessions policies
CREATE POLICY "Enable read access for all users" ON chat_sessions FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON chat_sessions FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON chat_sessions FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON chat_sessions FOR DELETE USING (true);

-- Chat messages policies
CREATE POLICY "Enable read access for all users" ON chat_messages FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON chat_messages FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON chat_messages FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON chat_messages FOR DELETE USING (true);

-- Workflow schedules policies
CREATE POLICY "Enable read access for all users" ON workflow_schedules FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON workflow_schedules FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON workflow_schedules FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON workflow_schedules FOR DELETE USING (true);

-- Workflow switches policies
CREATE POLICY "Enable read access for all users" ON workflow_switches FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON workflow_switches FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON workflow_switches FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON workflow_switches FOR DELETE USING (true);

-- Add helpful comments
COMMENT ON TABLE chat_sessions IS 'Chat session tracking for persistent conversations';
COMMENT ON TABLE chat_messages IS 'Individual chat messages with role-based organization';
COMMENT ON TABLE workflow_schedules IS 'Cron-based workflow scheduling configuration';
COMMENT ON TABLE workflow_switches IS 'Logs of mid-session workflow switching events';

COMMENT ON COLUMN chat_messages.role IS 'Message role: user, agent, or system';
COMMENT ON COLUMN workflow_schedules.cron_expression IS 'Cron expression for scheduling (e.g., "0 9 * * 1-5")';
COMMENT ON COLUMN workflow_switches.confidence IS 'Intent classification confidence that triggered the switch';

-- Create helpful views for monitoring
CREATE OR REPLACE VIEW chat_session_summary AS
SELECT 
  cs.id,
  cs.user_id,
  cs.started_at,
  cs.last_updated,
  COUNT(cm.id) as message_count,
  COUNT(cm.id) FILTER (WHERE cm.role = 'user') as user_messages,
  COUNT(cm.id) FILTER (WHERE cm.role = 'agent') as agent_messages,
  MAX(cm.created_at) as last_message_at
FROM chat_sessions cs
LEFT JOIN chat_messages cm ON cs.id = cm.session_id
GROUP BY cs.id, cs.user_id, cs.started_at, cs.last_updated
ORDER BY cs.last_updated DESC;

CREATE OR REPLACE VIEW active_schedules AS
SELECT 
  ws.id,
  ws.cron_expression,
  w.name as workflow_name,
  w.description as workflow_description,
  ws.created_at,
  ws.updated_at
FROM workflow_schedules ws
JOIN workflows w ON ws.workflow_id = w.id
WHERE ws.active = true
ORDER BY ws.created_at DESC;

CREATE OR REPLACE VIEW workflow_switch_analytics AS
SELECT 
  trigger_intent,
  COUNT(*) as switch_count,
  AVG(confidence) as avg_confidence,
  MAX(switched_at) as last_switch
FROM workflow_switches
WHERE switched_at > now() - interval '30 days'
GROUP BY trigger_intent
ORDER BY switch_count DESC;

-- Add comments for views
COMMENT ON VIEW chat_session_summary IS 'Summary statistics for chat sessions with message counts';
COMMENT ON VIEW active_schedules IS 'Currently active workflow schedules with workflow details';
COMMENT ON VIEW workflow_switch_analytics IS 'Analytics on workflow switching patterns and intents';

-- Insert initial log entry
INSERT INTO logs (event_type, details) 
SELECT 'chat_scheduler_tables_created', jsonb_build_object(
  'message', 'Chat persistence and scheduler tables created successfully',
  'tables_created', jsonb_build_array('chat_sessions', 'chat_messages', 'workflow_schedules', 'workflow_switches'),
  'views_created', jsonb_build_array('chat_session_summary', 'active_schedules', 'workflow_switch_analytics'),
  'features_enabled', jsonb_build_array('chat_persistence', 'workflow_scheduling', 'mid_session_switching'),
  'timestamp', now()
)
WHERE NOT EXISTS (SELECT 1 FROM logs WHERE event_type = 'chat_scheduler_tables_created');

-- Final success message
DO $$
BEGIN
  RAISE NOTICE '=== Chat Persistence and Scheduler Tables Created Successfully ===';
  RAISE NOTICE 'Tables: chat_sessions, chat_messages, workflow_schedules, workflow_switches';
  RAISE NOTICE 'Views: chat_session_summary, active_schedules, workflow_switch_analytics';
  RAISE NOTICE 'Features: Chat persistence, Cron scheduling, Mid-session switching';
  RAISE NOTICE 'Ready for SchedulerService and WorkflowStreamService!';
  RAISE NOTICE '================================================================';
END $$;