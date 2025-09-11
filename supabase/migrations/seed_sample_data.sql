/*
  # AgentFlow Sample Data Seed - Idempotent Version
  
  This migration inserts comprehensive sample data for the AgentFlow platform.
  It's designed to be completely idempotent and safe to run multiple times.
  
  1. Sample Tools
    - get_postpaid_balance - Retrieves customer account balance
    - authenticate_user - Handles user authentication
    - send_notification - Sends notifications to users
    - create_ticket - Creates customer support tickets

  2. Sample Agents
    - RouterAgent - Routes incoming requests based on intent analysis
    - AuthAgent - Handles authentication and session management
    - PostpaidBalanceAgent - Retrieves and presents balance information
    - StopAgent - Terminates workflows and provides final responses
    - SupportAgent - Handles customer support and ticket creation
    - NotificationAgent - Sends notifications and updates

  3. Sample Workflows
    - Customer Service Workflow - Complete customer service flow
    - Balance Check Flow - Streamlined balance checking
    - Support Ticket Flow - Support ticket creation and management

  All inserts use ON CONFLICT DO NOTHING for idempotency.
*/

-- Insert sample tools first (referenced by agents) - IDEMPOTENT
INSERT INTO tools (name, description, function_code, parameters) VALUES
(
  'get_postpaid_balance',
  'Retrieves the postpaid balance for a customer account',
  'function get_postpaid_balance(customer_id) { 
    return { 
      customer_id: customer_id, 
      balance: 85.50, 
      currency: "USD", 
      last_updated: new Date().toISOString(),
      account_status: "active",
      due_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString()
    }; 
  }',
  jsonb_build_object(
    'customer_id', jsonb_build_object(
      'type', 'string',
      'required', true,
      'description', 'Customer ID to retrieve balance for'
    )
  )
),
(
  'authenticate_user',
  'Authenticates user credentials and returns session information',
  'function authenticate_user(user_id, password) { 
    return { 
      user_id: user_id, 
      authenticated: true, 
      session_token: "mock_token_" + Date.now(),
      expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
      user_role: "customer"
    }; 
  }',
  jsonb_build_object(
    'user_id', jsonb_build_object(
      'type', 'string',
      'required', true,
      'description', 'User ID to authenticate'
    ),
    'password', jsonb_build_object(
      'type', 'string',
      'required', false,
      'description', 'User password (optional for demo)'
    )
  )
),
(
  'send_notification',
  'Sends a notification to a user',
  'function send_notification(recipient, message) { 
    return { 
      notification_id: "notif_" + Date.now(), 
      recipient: recipient,
      message: message,
      status: "sent",
      sent_at: new Date().toISOString()
    }; 
  }',
  jsonb_build_object(
    'recipient', jsonb_build_object(
      'type', 'string',
      'required', true,
      'description', 'Recipient user ID'
    ),
    'message', jsonb_build_object(
      'type', 'string',
      'required', true,
      'description', 'Notification message'
    )
  )
),
(
  'create_ticket',
  'Creates a support ticket for customer issues',
  'function create_ticket(customer_id, subject, priority) { 
    return { 
      ticket_id: "ticket_" + Date.now(), 
      customer_id: customer_id,
      subject: subject,
      status: "open",
      priority: priority || "medium",
      created_at: new Date().toISOString()
    }; 
  }',
  jsonb_build_object(
    'customer_id', jsonb_build_object(
      'type', 'string',
      'required', true,
      'description', 'Customer ID'
    ),
    'subject', jsonb_build_object(
      'type', 'string',
      'required', true,
      'description', 'Ticket subject'
    ),
    'priority', jsonb_build_object(
      'type', 'string',
      'required', false,
      'description', 'Ticket priority (low, medium, high)'
    )
  )
)
ON CONFLICT (name) DO NOTHING;

-- Insert sample agents with proper intent mapping - IDEMPOTENT
INSERT INTO agents (name, description, system_prompt, model_name, input_intents, output_intents, tool_id) VALUES
(
  'RouterAgent',
  'Routes incoming user requests to appropriate agents based on query analysis',
  'You are a router agent responsible for analyzing user queries and determining the appropriate workflow path. For customer service queries about account balance, billing, or account information, route to authentication. For general inquiries, provide helpful responses. Always be professional and helpful.',
  'gpt-4',
  '["user_query", "initial_request"]'::jsonb,
  '["need_authentication", "need_support", "workflow_complete"]'::jsonb,
  null
),
(
  'AuthAgent',
  'Handles user authentication and session management for secure operations',
  'You are an authentication agent responsible for verifying user identity and establishing secure sessions. Authenticate users using their credentials and create secure session tokens. Always prioritize security and data protection.',
  'gpt-4',
  '["need_authentication", "auth_required"]'::jsonb,
  '["need_balance_info", "auth_failed", "need_support"]'::jsonb,
  (SELECT id FROM tools WHERE name = 'authenticate_user')
),
(
  'PostpaidBalanceAgent',
  'Retrieves and presents postpaid account balance information',
  'You are a balance agent specialized in retrieving and presenting customer account balance information. Provide clear, accurate balance details including current balance, due dates, and account status. Always format financial information clearly.',
  'gpt-4',
  '["need_balance_info", "balance_request"]'::jsonb,
  '["workflow_complete", "need_support", "balance_retrieved"]'::jsonb,
  (SELECT id FROM tools WHERE name = 'get_postpaid_balance')
),
(
  'StopAgent',
  'Terminates workflow execution and provides final responses to users',
  'You are a stop agent responsible for concluding workflows and providing final responses to users. Summarize the completed actions, present results clearly, and ensure the user has received all requested information. Always end on a helpful and professional note.',
  'gpt-4',
  '["workflow_complete", "workflow_end", "final_response"]'::jsonb,
  '[]'::jsonb,
  null
),
(
  'SupportAgent',
  'Handles customer support requests and creates support tickets',
  'You are a customer support agent responsible for handling customer inquiries, troubleshooting issues, and creating support tickets when needed. Provide helpful solutions and escalate complex issues appropriately.',
  'gpt-4',
  '["need_support", "support_request", "create_ticket"]'::jsonb,
  '["workflow_complete", "ticket_created", "issue_resolved"]'::jsonb,
  (SELECT id FROM tools WHERE name = 'create_ticket')
),
(
  'NotificationAgent',
  'Sends notifications and updates to users',
  'You are a notification agent responsible for sending important updates, alerts, and notifications to users. Ensure messages are clear, timely, and relevant to the user context.',
  'gpt-4',
  '["send_notification", "notify_user"]'::jsonb,
  '["notification_sent", "workflow_complete"]'::jsonb,
  (SELECT id FROM tools WHERE name = 'send_notification')
)
ON CONFLICT (name) DO NOTHING;

-- Insert sample workflows - IDEMPOTENT
INSERT INTO workflows (name, description) VALUES
(
  'Customer Service Workflow',
  'Complete customer service workflow handling authentication, balance inquiries, and support requests'
),
(
  'Balance Check Flow',
  'Streamlined workflow for checking postpaid account balances'
),
(
  'Support Ticket Flow',
  'Workflow for creating and managing customer support tickets'
)
ON CONFLICT (name) DO NOTHING;

-- Create workflow nodes and edges for Customer Service Workflow - IDEMPOTENT
DO $$
DECLARE
  workflow_uuid uuid;
  router_node_id uuid;
  auth_node_id uuid;
  balance_node_id uuid;
  stop_node_id uuid;
  support_node_id uuid;
  notification_node_id uuid;
BEGIN
  -- Get the Customer Service Workflow ID
  SELECT id INTO workflow_uuid FROM workflows WHERE name = 'Customer Service Workflow';
  
  -- Only proceed if workflow exists and doesn't already have nodes
  IF workflow_uuid IS NOT NULL AND NOT EXISTS (SELECT 1 FROM workflow_nodes WHERE workflow_id = workflow_uuid) THEN
    
    -- Insert workflow nodes
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (workflow_uuid, (SELECT id FROM agents WHERE name = 'RouterAgent'), 'start', '{"x": 100, "y": 200}'::jsonb) 
    RETURNING id INTO router_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (workflow_uuid, (SELECT id FROM agents WHERE name = 'AuthAgent'), 'agent', '{"x": 300, "y": 150}'::jsonb) 
    RETURNING id INTO auth_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (workflow_uuid, (SELECT id FROM agents WHERE name = 'PostpaidBalanceAgent'), 'agent', '{"x": 500, "y": 100}'::jsonb) 
    RETURNING id INTO balance_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (workflow_uuid, (SELECT id FROM agents WHERE name = 'SupportAgent'), 'agent', '{"x": 500, "y": 250}'::jsonb) 
    RETURNING id INTO support_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (workflow_uuid, (SELECT id FROM agents WHERE name = 'NotificationAgent'), 'agent', '{"x": 700, "y": 150}'::jsonb) 
    RETURNING id INTO notification_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (workflow_uuid, (SELECT id FROM agents WHERE name = 'StopAgent'), 'end', '{"x": 900, "y": 200}'::jsonb) 
    RETURNING id INTO stop_node_id;
    
    -- Insert workflow edges (intent-based routing)
    INSERT INTO workflow_edges (workflow_id, from_node_id, to_node_id, trigger_intent) VALUES
    -- Router to Auth
    (workflow_uuid, router_node_id, auth_node_id, 'need_authentication'),
    -- Router to Support (for direct support requests)
    (workflow_uuid, router_node_id, support_node_id, 'need_support'),
    -- Auth to Balance (successful auth)
    (workflow_uuid, auth_node_id, balance_node_id, 'need_balance_info'),
    -- Auth to Support (auth failed)
    (workflow_uuid, auth_node_id, support_node_id, 'auth_failed'),
    -- Balance to Notification
    (workflow_uuid, balance_node_id, notification_node_id, 'balance_retrieved'),
    -- Balance to Stop (direct completion)
    (workflow_uuid, balance_node_id, stop_node_id, 'workflow_complete'),
    -- Support to Stop
    (workflow_uuid, support_node_id, stop_node_id, 'workflow_complete'),
    -- Support to Stop (ticket created)
    (workflow_uuid, support_node_id, stop_node_id, 'ticket_created'),
    -- Notification to Stop
    (workflow_uuid, notification_node_id, stop_node_id, 'notification_sent'),
    -- Notification to Stop (workflow complete)
    (workflow_uuid, notification_node_id, stop_node_id, 'workflow_complete');
    
  END IF;
END $$;

-- Create simplified Balance Check Flow - IDEMPOTENT
DO $$
DECLARE
  balance_workflow_uuid uuid;
  router_node_id uuid;
  auth_node_id uuid;
  balance_node_id uuid;
  stop_node_id uuid;
BEGIN
  -- Get the Balance Check Flow ID
  SELECT id INTO balance_workflow_uuid FROM workflows WHERE name = 'Balance Check Flow';
  
  -- Only proceed if workflow exists and doesn't already have nodes
  IF balance_workflow_uuid IS NOT NULL AND NOT EXISTS (SELECT 1 FROM workflow_nodes WHERE workflow_id = balance_workflow_uuid) THEN
    
    -- Insert workflow nodes for simplified flow
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (balance_workflow_uuid, (SELECT id FROM agents WHERE name = 'RouterAgent'), 'start', '{"x": 100, "y": 100}'::jsonb) 
    RETURNING id INTO router_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (balance_workflow_uuid, (SELECT id FROM agents WHERE name = 'AuthAgent'), 'agent', '{"x": 300, "y": 100}'::jsonb) 
    RETURNING id INTO auth_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (balance_workflow_uuid, (SELECT id FROM agents WHERE name = 'PostpaidBalanceAgent'), 'agent', '{"x": 500, "y": 100}'::jsonb) 
    RETURNING id INTO balance_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (balance_workflow_uuid, (SELECT id FROM agents WHERE name = 'StopAgent'), 'end', '{"x": 700, "y": 100}'::jsonb) 
    RETURNING id INTO stop_node_id;
    
    -- Insert workflow edges for simplified flow
    INSERT INTO workflow_edges (workflow_id, from_node_id, to_node_id, trigger_intent) VALUES
    (balance_workflow_uuid, router_node_id, auth_node_id, 'need_authentication'),
    (balance_workflow_uuid, auth_node_id, balance_node_id, 'need_balance_info'),
    (balance_workflow_uuid, balance_node_id, stop_node_id, 'workflow_complete');
    
  END IF;
END $$;

-- Create Support Ticket Flow - IDEMPOTENT
DO $$
DECLARE
  support_workflow_uuid uuid;
  router_node_id uuid;
  support_node_id uuid;
  notification_node_id uuid;
  stop_node_id uuid;
BEGIN
  -- Get the Support Ticket Flow ID
  SELECT id INTO support_workflow_uuid FROM workflows WHERE name = 'Support Ticket Flow';
  
  -- Only proceed if workflow exists and doesn't already have nodes
  IF support_workflow_uuid IS NOT NULL AND NOT EXISTS (SELECT 1 FROM workflow_nodes WHERE workflow_id = support_workflow_uuid) THEN
    
    -- Insert workflow nodes for support flow
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (support_workflow_uuid, (SELECT id FROM agents WHERE name = 'RouterAgent'), 'start', '{"x": 100, "y": 100}'::jsonb) 
    RETURNING id INTO router_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (support_workflow_uuid, (SELECT id FROM agents WHERE name = 'SupportAgent'), 'agent', '{"x": 300, "y": 100}'::jsonb) 
    RETURNING id INTO support_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (support_workflow_uuid, (SELECT id FROM agents WHERE name = 'NotificationAgent'), 'agent', '{"x": 500, "y": 100}'::jsonb) 
    RETURNING id INTO notification_node_id;
    
    INSERT INTO workflow_nodes (workflow_id, agent_id, node_type, position) VALUES
    (support_workflow_uuid, (SELECT id FROM agents WHERE name = 'StopAgent'), 'end', '{"x": 700, "y": 100}'::jsonb) 
    RETURNING id INTO stop_node_id;
    
    -- Insert workflow edges for support flow
    INSERT INTO workflow_edges (workflow_id, from_node_id, to_node_id, trigger_intent) VALUES
    (support_workflow_uuid, router_node_id, support_node_id, 'need_support'),
    (support_workflow_uuid, support_node_id, notification_node_id, 'ticket_created'),
    (support_workflow_uuid, notification_node_id, stop_node_id, 'notification_sent');
    
  END IF;
END $$;

-- Insert sample logs using proper JSONB construction - IDEMPOTENT
INSERT INTO logs (event_type, details) 
SELECT 'system_startup', jsonb_build_object(
  'message', 'AgentFlow system initialized successfully',
  'agents_loaded', 6,
  'tools_registered', 4,
  'workflows_available', 3,
  'timestamp', '2025-06-28T10:00:00Z',
  'version', '1.0.0',
  'environment', 'development'
)
WHERE NOT EXISTS (SELECT 1 FROM logs WHERE event_type = 'system_startup');

INSERT INTO logs (event_type, details) 
SELECT 'seed_data_loaded', jsonb_build_object(
  'message', 'Sample data successfully loaded into database',
  'workflows', jsonb_build_array('Customer Service Workflow', 'Balance Check Flow', 'Support Ticket Flow'),
  'agents', jsonb_build_array('RouterAgent', 'AuthAgent', 'PostpaidBalanceAgent', 'StopAgent', 'SupportAgent', 'NotificationAgent'),
  'tools', jsonb_build_array('get_postpaid_balance', 'authenticate_user', 'send_notification', 'create_ticket'),
  'timestamp', '2025-06-28T10:01:00Z',
  'total_nodes', 16,
  'total_edges', 16
)
WHERE NOT EXISTS (SELECT 1 FROM logs WHERE event_type = 'seed_data_loaded');

INSERT INTO logs (event_type, details) 
SELECT 'database_ready', jsonb_build_object(
  'message', 'AgentFlow database is ready for workflow execution',
  'schema_version', '1.0',
  'tables_created', jsonb_build_array('tools', 'agents', 'workflows', 'workflow_nodes', 'workflow_edges', 'workflow_runs', 'logs'),
  'views_created', jsonb_build_array('workflow_summary', 'agent_tool_summary', 'recent_workflow_runs'),
  'timestamp', '2025-06-28T10:02:00Z',
  'rls_enabled', true,
  'indexes_created', 20
)
WHERE NOT EXISTS (SELECT 1 FROM logs WHERE event_type = 'database_ready');

-- Insert a sample workflow run for demonstration - IDEMPOTENT
DO $$
DECLARE
  sample_workflow_id uuid;
  sample_run_id uuid;
BEGIN
  -- Get a workflow ID for the sample run
  SELECT id INTO sample_workflow_id FROM workflows WHERE name = 'Balance Check Flow' LIMIT 1;
  
  -- Only insert if no sample run exists
  IF sample_workflow_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM workflow_runs 
    WHERE workflow_id = sample_workflow_id 
    AND input_data->>'user_id' = 'demo_user_123'
  ) THEN
    
    -- Insert a sample workflow run
    INSERT INTO workflow_runs (workflow_id, status, input_data, output_data, execution_path, started_at, completed_at)
    VALUES (
      sample_workflow_id,
      'completed',
      jsonb_build_object(
        'query', 'Check my account balance',
        'user_id', 'demo_user_123',
        'customer_name', 'krishna'
      ),
      jsonb_build_object(
        'message', 'Balance retrieved successfully',
        'customer_id', 'demo_user_123',
        'balance', 85.50,
        'currency', 'USD',
        'account_status', 'active'
      ),
      jsonb_build_array(
        jsonb_build_object(
          'step', 1,
          'agent_name', 'RouterAgent',
          'input', jsonb_build_object('query', 'Check my account balance'),
          'output', jsonb_build_object('routing_decision', 'authenticate_user'),
          'output_intent', 'need_authentication',
          'timestamp', '2025-06-28T10:05:00Z'
        ),
        jsonb_build_object(
          'step', 2,
          'agent_name', 'AuthAgent',
          'input', jsonb_build_object('user_id', 'demo_user_123'),
          'output', jsonb_build_object('authenticated', true, 'session_token', 'demo_token_123'),
          'output_intent', 'need_balance_info',
          'timestamp', '2025-06-28T10:05:01Z'
        ),
        jsonb_build_object(
          'step', 3,
          'agent_name', 'PostpaidBalanceAgent',
          'input', jsonb_build_object('customer_id', 'demo_user_123'),
          'output', jsonb_build_object('balance', 85.50, 'currency', 'USD'),
          'output_intent', 'workflow_complete',
          'timestamp', '2025-06-28T10:05:02Z'
        )
      ),
      '2025-06-28T10:05:00Z'::timestamptz,
      '2025-06-28T10:05:03Z'::timestamptz
    ) RETURNING id INTO sample_run_id;
    
    -- Insert corresponding log entries for the sample run
    INSERT INTO logs (event_type, workflow_run_id, details) VALUES
    (
      'workflow_start',
      sample_run_id,
      jsonb_build_object(
        'workflow_name', 'Balance Check Flow',
        'input_data', jsonb_build_object('query', 'Check my account balance'),
        'timestamp', '2025-06-28T10:05:00Z'
      )
    ),
    (
      'workflow_complete',
      sample_run_id,
      jsonb_build_object(
        'workflow_name', 'Balance Check Flow',
        'total_steps', 3,
        'execution_time_seconds', 3,
        'final_output', jsonb_build_object('balance', 85.50, 'currency', 'USD'),
        'timestamp', '2025-06-28T10:05:03Z'
      )
    );
    
  END IF;
END $$;

-- Final verification and success message
DO $$
DECLARE
  tool_count integer;
  agent_count integer;
  workflow_count integer;
  node_count integer;
  edge_count integer;
  log_count integer;
BEGIN
  SELECT COUNT(*) INTO tool_count FROM tools;
  SELECT COUNT(*) INTO agent_count FROM agents;
  SELECT COUNT(*) INTO workflow_count FROM workflows;
  SELECT COUNT(*) INTO node_count FROM workflow_nodes;
  SELECT COUNT(*) INTO edge_count FROM workflow_edges;
  SELECT COUNT(*) INTO log_count FROM logs;
  
  RAISE NOTICE '=== AgentFlow Seed Data Successfully Loaded ===';
  RAISE NOTICE 'Tools created: %', tool_count;
  RAISE NOTICE 'Agents created: %', agent_count;
  RAISE NOTICE 'Workflows created: %', workflow_count;
  RAISE NOTICE 'Workflow nodes created: %', node_count;
  RAISE NOTICE 'Workflow edges created: %', edge_count;
  RAISE NOTICE 'Log entries created: %', log_count;
  RAISE NOTICE '';
  RAISE NOTICE 'Sample workflows available:';
  RAISE NOTICE '1. Customer Service Workflow (6 nodes, 10 edges)';
  RAISE NOTICE '2. Balance Check Flow (4 nodes, 3 edges)';
  RAISE NOTICE '3. Support Ticket Flow (4 nodes, 3 edges)';
  RAISE NOTICE '';
  RAISE NOTICE 'Ready to execute: node server.js && node scripts/test_workflow.js';
  RAISE NOTICE '================================================';
END $$;