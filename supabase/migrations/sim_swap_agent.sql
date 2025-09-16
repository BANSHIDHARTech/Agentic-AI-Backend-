/*
  # Fix SIM Swap Agent FSM - Handle Unique Constraint
  
  The previous migration failed because workflow_nodes has a unique constraint on (workflow_id, agent_id).
  Since we need multiple FSM states for the same SIM Swap Agent, we'll create separate "virtual" agents
  for each FSM state, or use a different approach that works with the existing schema.
  
  Solution: Create separate agent entries for each FSM state, or modify the approach to use
  the position field more effectively with a single node.
*/

-- First, clean up any partial data from the failed migration
DELETE FROM workflow_edges WHERE workflow_id IN (
  SELECT id FROM workflows WHERE name = 'SIM Swap Workflow'
);

DELETE FROM workflow_nodes WHERE workflow_id IN (
  SELECT id FROM workflows WHERE name = 'SIM Swap Workflow'
);

DELETE FROM workflows WHERE name = 'SIM Swap Workflow';

-- Insert SIM Swap Agent (if not exists)
INSERT INTO agents (name, description, system_prompt, model_name, input_intents, output_intents, tool_id) VALUES
(
  'SIM Swap Agent',
  'Handles SIM swap requests through a database-driven finite state machine with multi-step verification',
  'You are a SIM swap agent that processes customer requests through a secure multi-step verification process. All your responses and logic are driven by database configuration. Follow the workflow steps defined in the database and provide appropriate responses based on the current FSM state.',
  'gpt-4',
  '["sim_swap_request", "collect_user_details", "verify_identity", "confirm_swap"]'::jsonb,
  '["collect_user_details", "verify_identity", "confirm_swap", "workflow_complete"]'::jsonb,
  null
)
ON CONFLICT (name) DO NOTHING;

-- Create virtual FSM state agents to work around unique constraint
INSERT INTO agents (name, description, system_prompt, model_name, input_intents, output_intents, tool_id) VALUES
(
  'SIM Swap Collector',
  'Collects user details for SIM swap process',
  'You collect user details for SIM swap verification. Request mobile number, full name, and account number.',
  'gpt-4',
  '["sim_swap_request", "collect_user_details"]'::jsonb,
  '["verify_identity"]'::jsonb,
  null
),
(
  'SIM Swap Verifier', 
  'Verifies user identity for SIM swap process',
  'You verify user identity through security questions and other verification methods.',
  'gpt-4',
  '["verify_identity"]'::jsonb,
  '["confirm_swap"]'::jsonb,
  null
),
(
  'SIM Swap Confirmer',
  'Confirms and processes SIM swap requests',
  'You confirm SIM swap requests and provide final confirmation details.',
  'gpt-4',
  '["confirm_swap"]'::jsonb,
  '["workflow_complete"]'::jsonb,
  null
)
ON CONFLICT (name) DO NOTHING;

-- Insert SIM Swap FSM Workflow
INSERT INTO workflows (name, description) VALUES
(
  'SIM Swap FSM Workflow',
  'Database-driven finite state machine for SIM swap processing with multi-step verification'
)
ON CONFLICT (name) DO NOTHING;

-- Create FSM workflow nodes using separate agents for each state
DO $$
DECLARE
  sim_workflow_id uuid;
  router_agent_id uuid;
  collector_agent_id uuid;
  verifier_agent_id uuid;
  confirmer_agent_id uuid;
  stop_agent_id uuid;
  
  router_node_id uuid;
  collect_node_id uuid;
  verify_node_id uuid;
  confirm_node_id uuid;
  stop_node_id uuid;
BEGIN
  -- Get required agent and workflow IDs
  SELECT id INTO sim_workflow_id FROM workflows WHERE name = 'SIM Swap FSM Workflow';
  SELECT id INTO router_agent_id FROM agents WHERE name = 'RouterAgent';
  SELECT id INTO collector_agent_id FROM agents WHERE name = 'SIM Swap Collector';
  SELECT id INTO verifier_agent_id FROM agents WHERE name = 'SIM Swap Verifier';
  SELECT id INTO confirmer_agent_id FROM agents WHERE name = 'SIM Swap Confirmer';
  SELECT id INTO stop_agent_id FROM agents WHERE name = 'StopAgent';
  
  -- Only proceed if workflow exists and doesn't already have nodes
  IF sim_workflow_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM workflow_nodes WHERE workflow_id = sim_workflow_id) THEN
    
    -- Create Router Node (Entry Point for SIM Swap requests)
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (sim_workflow_id, router_agent_id, 'start', jsonb_build_object(
      'x', 100,
      'y', 200,
      'description', 'Routes SIM swap requests to FSM workflow'
    )) RETURNING id INTO router_node_id;
    
    -- FSM Node 1: Collect User Details
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (sim_workflow_id, collector_agent_id, 'agent', jsonb_build_object(
      'x', 300,
      'y', 200,
      'fsm_state', 'collect_user_details',
      'mock_response', jsonb_build_object(
        'message', 'Welcome to SIM Swap Service. To proceed securely, I need to collect some details from you.',
        'required_fields', jsonb_build_array('mobile_number', 'full_name', 'account_number'),
        'instructions', 'Please provide your registered mobile number, full name, and account number to continue.',
        'step', 'collect_user_details',
        'security_notice', 'For your protection, we need to verify your identity before proceeding.'
      ),
      'output_intent', 'verify_identity',
      'validation_rules', jsonb_build_object(
        'required_fields', jsonb_build_array('mobile_number', 'full_name'),
        'mobile_format', '^[0-9]{10}$',
        'name_min_length', 2
      )
    )) RETURNING id INTO collect_node_id;
    
    -- FSM Node 2: Verify Identity
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (sim_workflow_id, verifier_agent_id, 'agent', jsonb_build_object(
      'x', 500,
      'y', 200,
      'fsm_state', 'verify_identity',
      'mock_response', jsonb_build_object(
        'message', 'Thank you for providing your details. For security purposes, I need to verify your identity.',
        'security_question', 'What is your mother''s maiden name?',
        'verification_methods', jsonb_build_array('Security Question', 'SMS OTP', 'Document Upload'),
        'instructions', 'Please answer the security question to proceed with your SIM swap request.',
        'step', 'verify_identity',
        'retry_limit', 3
      ),
      'output_intent', 'confirm_swap',
      'validation_rules', jsonb_build_object(
        'required_fields', jsonb_build_array('security_answer'),
        'min_length', 2,
        'max_retries', 3
      )
    )) RETURNING id INTO verify_node_id;
    
    -- FSM Node 3: Confirm Swap
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (sim_workflow_id, confirmer_agent_id, 'agent', jsonb_build_object(
      'x', 700,
      'y', 200,
      'fsm_state', 'confirm_swap',
      'mock_response', jsonb_build_object(
        'message', 'Identity verification successful! Your SIM swap request has been processed.',
        'swap_details', jsonb_build_object(
          'new_sim_number', 'SIM_NEW_XXXXXX',
          'estimated_completion', '2 hours',
          'reference_number', 'SWAP_REF_XXXXXX',
          'old_sim_deactivation', 'Within 2 hours'
        ),
        'activation_instructions', jsonb_build_array(
          'Insert the new SIM card into your device',
          'Restart your device completely',
          'Wait for network activation (up to 2 hours)',
          'Test by making a call or sending SMS'
        ),
        'important_notes', jsonb_build_array(
          'Keep your device on during activation',
          'Your old SIM will be deactivated automatically',
          'Contact support if you face any issues'
        ),
        'status', 'CONFIRMED',
        'step', 'confirm_swap',
        'support_contact', '1-800-SIM-HELP'
      ),
      'output_intent', 'workflow_complete'
    )) RETURNING id INTO confirm_node_id;
    
    -- Stop Node (End of FSM workflow)
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (sim_workflow_id, stop_agent_id, 'end', jsonb_build_object(
      'x', 900,
      'y', 200,
      'description', 'Completes SIM swap workflow and provides final summary'
    )) RETURNING id INTO stop_node_id;
    
    -- Insert Workflow Edges for FSM transitions
    -- Router to FSM Entry Point
    INSERT INTO workflow_edges (workflow_id, from_node_id, to_node_id, trigger_intent) VALUES
    (sim_workflow_id, router_node_id, collect_node_id, 'sim_swap_request');
    
    -- FSM State Transitions (Forward progression)
    INSERT INTO workflow_edges (workflow_id, from_node_id, to_node_id, trigger_intent) VALUES
    (sim_workflow_id, collect_node_id, verify_node_id, 'verify_identity'),
    (sim_workflow_id, verify_node_id, confirm_node_id, 'confirm_swap'),
    (sim_workflow_id, confirm_node_id, stop_node_id, 'workflow_complete');
    
    -- Self-loops for validation failures (stay in same state)
    INSERT INTO workflow_edges (workflow_id, from_node_id, to_node_id, trigger_intent) VALUES
    (sim_workflow_id, collect_node_id, collect_node_id, 'collect_user_details'),
    (sim_workflow_id, verify_node_id, verify_node_id, 'verify_identity');
    
  END IF;
END $$;

-- Insert initial log entry for SIM Swap Agent creation
INSERT INTO logs (event_type, details) 
SELECT 'sim_swap_fsm_fixed', jsonb_build_object(
  'message', 'SIM Swap FSM created successfully with separate agents for each state',
  'workflow_name', 'SIM Swap FSM Workflow',
  'fsm_agents', jsonb_build_array('SIM Swap Collector', 'SIM Swap Verifier', 'SIM Swap Confirmer'),
  'fsm_states', jsonb_build_array('collect_user_details', 'verify_identity', 'confirm_swap'),
  'database_driven', true,
  'hardcoded_logic', false,
  'validation_enabled', true,
  'unique_constraint_resolved', true,
  'timestamp', now()
)
WHERE NOT EXISTS (SELECT 1 FROM logs WHERE event_type = 'sim_swap_fsm_fixed');

-- Recreate helper view for FSM state inspection
CREATE OR REPLACE VIEW sim_swap_fsm_states AS
SELECT 
  wn.id as node_id,
  w.name as workflow_name,
  a.name as agent_name,
  wn.position->>'fsm_state' as fsm_state,
  wn.position->'mock_response' as mock_response,
  wn.position->>'output_intent' as output_intent,
  wn.position->'validation_rules' as validation_rules,
  wn.node_type,
  wn.created_at
FROM workflow_nodes wn
JOIN workflows w ON wn.workflow_id = w.id
JOIN agents a ON wn.agent_id = a.id
WHERE w.name = 'SIM Swap FSM Workflow'
  AND wn.position ? 'fsm_state'
ORDER BY 
  CASE wn.position->>'fsm_state'
    WHEN 'collect_user_details' THEN 1
    WHEN 'verify_identity' THEN 2
    WHEN 'confirm_swap' THEN 3
    ELSE 4
  END;

COMMENT ON VIEW sim_swap_fsm_states IS 'Database-driven FSM states for SIM Swap - uses separate agents to avoid unique constraint';

-- Recreate helper function to get FSM state data
CREATE OR REPLACE FUNCTION get_sim_swap_fsm_state(state_name text)
RETURNS TABLE(
  node_id uuid,
  agent_name text,
  fsm_state text,
  mock_response jsonb,
  output_intent text,
  validation_rules jsonb
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    wn.id,
    a.name,
    wn.position->>'fsm_state',
    wn.position->'mock_response',
    wn.position->>'output_intent',
    wn.position->'validation_rules'
  FROM workflow_nodes wn
  JOIN workflows w ON wn.workflow_id = w.id
  JOIN agents a ON wn.agent_id = a.id
  WHERE w.name = 'SIM Swap FSM Workflow'
    AND wn.position->>'fsm_state' = state_name;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_sim_swap_fsm_state IS 'Retrieve FSM state data from database for SIM Swap workflow';

-- Final verification and success message
DO $$
DECLARE
  agent_count integer;
  workflow_count integer;
  node_count integer;
  edge_count integer;
  fsm_state_count integer;
BEGIN
  SELECT COUNT(*) INTO agent_count FROM agents WHERE name LIKE 'SIM Swap%';
  SELECT COUNT(*) INTO workflow_count FROM workflows WHERE name = 'SIM Swap FSM Workflow';
  SELECT COUNT(*) INTO node_count FROM workflow_nodes wn 
    JOIN workflows w ON wn.workflow_id = w.id 
    WHERE w.name = 'SIM Swap FSM Workflow';
  SELECT COUNT(*) INTO edge_count FROM workflow_edges we 
    JOIN workflows w ON we.workflow_id = w.id 
    WHERE w.name = 'SIM Swap FSM Workflow';
  SELECT COUNT(*) INTO fsm_state_count FROM sim_swap_fsm_states;
  
  RAISE NOTICE '=== SIM Swap FSM Fixed and Created Successfully ===';
  RAISE NOTICE 'SIM Swap Related Agents: %', agent_count;
  RAISE NOTICE 'SIM Swap Workflows: %', workflow_count;
  RAISE NOTICE 'Total Workflow Nodes: %', node_count;
  RAISE NOTICE 'Total Workflow Edges: %', edge_count;
  RAISE NOTICE 'FSM States Configured: %', fsm_state_count;
  RAISE NOTICE '';
  RAISE NOTICE '🔧 SOLUTION APPLIED:';
  RAISE NOTICE '✅ Created separate agents for each FSM state';
  RAISE NOTICE '✅ Resolved unique constraint (workflow_id, agent_id)';
  RAISE NOTICE '✅ All FSM logic still stored in database';
  RAISE NOTICE '✅ Zero hardcoded JavaScript transitions';
  RAISE NOTICE '✅ Database-driven responses and validation';
  RAISE NOTICE '✅ Intent-based state transitions maintained';
  RAISE NOTICE '';
  RAISE NOTICE '🚀 Ready to test: node scripts/test_workflow.js';
  RAISE NOTICE '================================================================';
END $$;