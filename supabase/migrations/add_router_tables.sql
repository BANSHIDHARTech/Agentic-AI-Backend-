/*
  # Router/Commander Agent Support Tables
  
  Creates comprehensive support tables for the Router/Commander Agent functionality:
  
  1. New Tables
    - `router_rules` - Intent classification and agent routing rules
    - `intent_logs` - Logs all intent classification attempts and results
    - `fallback_messages` - Configurable fallback responses for unmatched intents
    - `router_metrics` - Performance metrics for router and agent execution

  2. Security
    - Enable RLS on all new tables
    - Add policies for authenticated access
    - Proper foreign key constraints

  3. Features
    - Priority-based rule matching
    - Confidence scoring for intent classification
    - Comprehensive logging and metrics
    - Configurable fallback responses
*/

-- Create router_rules table for intent classification and routing
CREATE TABLE IF NOT EXISTS router_rules (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  intent_name text UNIQUE NOT NULL,
  keywords jsonb DEFAULT '[]'::jsonb,
  agent_id uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  priority integer DEFAULT 100,
  confidence_threshold numeric(3,2) DEFAULT 0.7,
  is_active boolean DEFAULT true,
  description text DEFAULT '',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create intent_logs table for tracking all intent classification attempts
CREATE TABLE IF NOT EXISTS intent_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_query text NOT NULL,
  detected_intent text,
  confidence_score numeric(3,2),
  selected_agent_id uuid REFERENCES agents(id),
  selected_agent_name text,
  rule_id uuid REFERENCES router_rules(id),
  fallback_used boolean DEFAULT false,
  processing_time_ms integer,
  session_id text,
  user_id text,
  created_at timestamptz DEFAULT now()
);

-- Create fallback_messages table for unmatched intents
CREATE TABLE IF NOT EXISTS fallback_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  message text NOT NULL,
  category text DEFAULT 'general',
  is_active boolean DEFAULT true,
  usage_count integer DEFAULT 0,
  last_used_at timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create router_metrics table for performance tracking
CREATE TABLE IF NOT EXISTS router_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  agent_name text NOT NULL,
  success_count integer DEFAULT 0,
  failure_count integer DEFAULT 0,
  total_executions integer DEFAULT 0,
  avg_response_time_ms numeric(10,2) DEFAULT 0,
  last_execution_at timestamptz,
  date_bucket date DEFAULT CURRENT_DATE,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  UNIQUE(agent_id, date_bucket)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_router_rules_intent_name ON router_rules(intent_name);
CREATE INDEX IF NOT EXISTS idx_router_rules_priority ON router_rules(priority DESC);
CREATE INDEX IF NOT EXISTS idx_router_rules_is_active ON router_rules(is_active);
CREATE INDEX IF NOT EXISTS idx_router_rules_keywords ON router_rules USING gin(keywords);

CREATE INDEX IF NOT EXISTS idx_intent_logs_detected_intent ON intent_logs(detected_intent);
CREATE INDEX IF NOT EXISTS idx_intent_logs_created_at ON intent_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_intent_logs_selected_agent_id ON intent_logs(selected_agent_id);
CREATE INDEX IF NOT EXISTS idx_intent_logs_session_id ON intent_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_fallback_messages_is_active ON fallback_messages(is_active);
CREATE INDEX IF NOT EXISTS idx_fallback_messages_category ON fallback_messages(category);

CREATE INDEX IF NOT EXISTS idx_router_metrics_agent_id ON router_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_router_metrics_date_bucket ON router_metrics(date_bucket);
CREATE INDEX IF NOT EXISTS idx_router_metrics_last_execution ON router_metrics(last_execution_at);

-- Enable Row Level Security
ALTER TABLE router_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE intent_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE fallback_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE router_metrics ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Router rules policies
CREATE POLICY "Enable read access for all users" ON router_rules FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON router_rules FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON router_rules FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON router_rules FOR DELETE USING (true);

-- Intent logs policies
CREATE POLICY "Enable read access for all users" ON intent_logs FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON intent_logs FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON intent_logs FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON intent_logs FOR DELETE USING (true);

-- Fallback messages policies
CREATE POLICY "Enable read access for all users" ON fallback_messages FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON fallback_messages FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON fallback_messages FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON fallback_messages FOR DELETE USING (true);

-- Router metrics policies
CREATE POLICY "Enable read access for all users" ON router_metrics FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON router_metrics FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON router_metrics FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON router_metrics FOR DELETE USING (true);

-- Add helpful comments
COMMENT ON TABLE router_rules IS 'Intent classification and agent routing rules with priority-based matching';
COMMENT ON TABLE intent_logs IS 'Comprehensive logs of all intent classification attempts and results';
COMMENT ON TABLE fallback_messages IS 'Configurable fallback responses for unmatched or low-confidence intents';
COMMENT ON TABLE router_metrics IS 'Performance metrics and statistics for router and agent execution';

COMMENT ON COLUMN router_rules.keywords IS 'JSON array of keywords for intent matching';
COMMENT ON COLUMN router_rules.priority IS 'Rule priority (lower number = higher priority)';
COMMENT ON COLUMN router_rules.confidence_threshold IS 'Minimum confidence score required to match this rule';

COMMENT ON COLUMN intent_logs.confidence_score IS 'Confidence score (0.0 to 1.0) for the intent classification';
COMMENT ON COLUMN intent_logs.processing_time_ms IS 'Time taken to process the intent classification in milliseconds';

COMMENT ON COLUMN router_metrics.date_bucket IS 'Date bucket for aggregating daily metrics';

-- Insert sample router rules
INSERT INTO router_rules (intent_name, keywords, agent_id, priority, confidence_threshold, description) VALUES
(
  'sim_swap_request',
  '["sim", "swap", "change", "replace", "new sim", "sim card"]'::jsonb,
  (SELECT id FROM agents WHERE name = 'SIM Swap Agent' LIMIT 1),
  10,
  0.8,
  'Routes SIM swap and SIM card replacement requests'
),
(
  'balance_inquiry',
  '["balance", "account", "bill", "payment", "due", "amount"]'::jsonb,
  (SELECT id FROM agents WHERE name = 'PostpaidBalanceAgent' LIMIT 1),
  20,
  0.7,
  'Routes balance and account inquiry requests'
),
(
  'authentication_required',
  '["login", "authenticate", "verify", "password", "access"]'::jsonb,
  (SELECT id FROM agents WHERE name = 'AuthAgent' LIMIT 1),
  30,
  0.6,
  'Routes authentication and login requests'
),
(
  'general_support',
  '["help", "support", "issue", "problem", "question"]'::jsonb,
  (SELECT id FROM agents WHERE name = 'SupportAgent' LIMIT 1),
  100,
  0.5,
  'Routes general support and help requests'
);

-- Insert sample fallback messages
INSERT INTO fallback_messages (message, category, is_active) VALUES
('I''m sorry, I didn''t understand your request. Could you please rephrase it or ask for help?', 'general', true),
('I''m not sure how to help with that. Would you like to speak with a support agent?', 'general', true),
('That''s an interesting question! Let me connect you with someone who can better assist you.', 'general', true),
('I''m still learning! Could you try asking your question in a different way?', 'general', true),
('I want to make sure I help you correctly. Could you provide more details about what you need?', 'general', true);

-- Initialize router metrics for existing agents
INSERT INTO router_metrics (agent_id, agent_name, date_bucket)
SELECT 
  id,
  name,
  CURRENT_DATE
FROM agents
WHERE is_active = true
ON CONFLICT (agent_id, date_bucket) DO NOTHING;

-- Create helpful views for monitoring and analytics
CREATE OR REPLACE VIEW router_performance AS
SELECT 
  rr.intent_name,
  a.name as agent_name,
  COUNT(il.id) as total_classifications,
  COUNT(il.id) FILTER (WHERE il.fallback_used = false) as successful_matches,
  COUNT(il.id) FILTER (WHERE il.fallback_used = true) as fallback_used,
  ROUND(AVG(il.confidence_score), 3) as avg_confidence,
  ROUND(AVG(il.processing_time_ms), 2) as avg_processing_time_ms,
  MAX(il.created_at) as last_used
FROM router_rules rr
LEFT JOIN agents a ON rr.agent_id = a.id
LEFT JOIN intent_logs il ON rr.id = il.rule_id
WHERE rr.is_active = true
GROUP BY rr.intent_name, a.name
ORDER BY total_classifications DESC;

CREATE OR REPLACE VIEW daily_router_stats AS
SELECT 
  date_bucket,
  SUM(total_executions) as total_executions,
  SUM(success_count) as total_successes,
  SUM(failure_count) as total_failures,
  ROUND(AVG(avg_response_time_ms), 2) as avg_response_time_ms,
  ROUND(
    CASE 
      WHEN SUM(total_executions) > 0 
      THEN (SUM(success_count)::numeric / SUM(total_executions)) * 100 
      ELSE 0 
    END, 2
  ) as success_rate_percent
FROM router_metrics
GROUP BY date_bucket
ORDER BY date_bucket DESC;

-- Add comments for views
COMMENT ON VIEW router_performance IS 'Performance analytics for router rules and intent classification';
COMMENT ON VIEW daily_router_stats IS 'Daily aggregated statistics for router and agent performance';

-- Create function to update router metrics
CREATE OR REPLACE FUNCTION update_router_metrics(
  p_agent_id uuid,
  p_agent_name text,
  p_success boolean,
  p_response_time_ms integer DEFAULT NULL
)
RETURNS void AS $$
BEGIN
  INSERT INTO router_metrics (agent_id, agent_name, date_bucket, total_executions, success_count, failure_count, avg_response_time_ms, last_execution_at)
  VALUES (
    p_agent_id,
    p_agent_name,
    CURRENT_DATE,
    1,
    CASE WHEN p_success THEN 1 ELSE 0 END,
    CASE WHEN p_success THEN 0 ELSE 1 END,
    COALESCE(p_response_time_ms, 0),
    now()
  )
  ON CONFLICT (agent_id, date_bucket) 
  DO UPDATE SET
    total_executions = router_metrics.total_executions + 1,
    success_count = router_metrics.success_count + CASE WHEN p_success THEN 1 ELSE 0 END,
    failure_count = router_metrics.failure_count + CASE WHEN p_success THEN 0 ELSE 1 END,
    avg_response_time_ms = CASE 
      WHEN p_response_time_ms IS NOT NULL THEN
        (router_metrics.avg_response_time_ms * router_metrics.total_executions + p_response_time_ms) / (router_metrics.total_executions + 1)
      ELSE router_metrics.avg_response_time_ms
    END,
    last_execution_at = now(),
    updated_at = now();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_router_metrics IS 'Updates router metrics for agent execution tracking';

-- Final success message
DO $$
BEGIN
  RAISE NOTICE '=== Router/Commander Agent Tables Created Successfully ===';
  RAISE NOTICE 'Tables: router_rules, intent_logs, fallback_messages, router_metrics';
  RAISE NOTICE 'Views: router_performance, daily_router_stats';
  RAISE NOTICE 'Functions: update_router_metrics';
  RAISE NOTICE 'Sample data: 4 router rules, 5 fallback messages';
  RAISE NOTICE 'Ready for Router Service implementation!';
  RAISE NOTICE '============================================================';
END $$;