-- Migration to add nodes and edges columns to the workflows table
-- This allows direct storage of nodes and edges in the workflows table
-- instead of relying on separate tables and sync functions

-- First add the columns if they don't exist
ALTER TABLE IF EXISTS workflows
ADD COLUMN IF NOT EXISTS nodes JSONB NOT NULL DEFAULT '[]',
ADD COLUMN IF NOT EXISTS edges JSONB NOT NULL DEFAULT '[]',
ADD COLUMN IF NOT EXISTS data JSONB DEFAULT NULL;

-- Update the workflows table with any existing nodes and edges
-- This will consolidate data from workflow_nodes and workflow_edges tables
DO $$
DECLARE
  wf RECORD;
  nodes_json JSONB;
  edges_json JSONB;
BEGIN
  -- For each workflow
  FOR wf IN SELECT id FROM workflows LOOP
    -- Get nodes for this workflow
    SELECT COALESCE(jsonb_agg(
      jsonb_build_object(
        'id', n.id,
        'type', n.node_type,
        'data', n.data,
        'position', n.position
      )
    ), '[]'::jsonb)
    INTO nodes_json
    FROM workflow_nodes n
    WHERE n.workflow_id = wf.id;
    
    -- Get edges for this workflow
    SELECT COALESCE(jsonb_agg(
      jsonb_build_object(
        'id', e.id,
        'source', e.from_node_id,
        'target', e.to_node_id,
        'data', jsonb_build_object(
          'label', e.trigger_intent,
          'condition', e.condition
        )
      )
    ), '[]'::jsonb)
    INTO edges_json
    FROM workflow_edges e
    WHERE e.workflow_id = wf.id;
    
    -- Update the workflow with nodes and edges
    UPDATE workflows
    SET 
      nodes = nodes_json,
      edges = edges_json,
      updated_at = now()
    WHERE id = wf.id;
  END LOOP;
END;
$$;

-- Now refresh the schema cache for PostgREST
-- This is required for PostgREST to recognize the new columns
NOTIFY pgrst, 'reload schema';

-- Log migration
INSERT INTO logs (event_type, details)
VALUES (
  'schema_migration',
  jsonb_build_object(
    'description', 'Added nodes and edges columns to workflows table',
    'timestamp', now()
  )
);