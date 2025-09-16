-- Create or replace sync_workflow function with transaction handling and trigger_intent fix
CREATE OR REPLACE FUNCTION sync_workflow(
  p_workflow_id uuid,
  p_nodes jsonb,
  p_edges jsonb
) RETURNS void AS $$
DECLARE
  v_workflow_exists boolean;
  v_node_count integer := 0;
  v_edge_count integer := 0;
  v_nodes jsonb;
  v_edges jsonb;
BEGIN
  -- Begin transaction
  -- Transaction is now explicitly declared at the function level
  
  -- Verify workflow exists (using LIMIT 1 to avoid multiple rows)
  SELECT EXISTS (SELECT 1 FROM workflows WHERE id = p_workflow_id LIMIT 1) INTO v_workflow_exists;
  IF NOT v_workflow_exists THEN
    RAISE EXCEPTION 'Workflow % does not exist', p_workflow_id;
  END IF;

  -- Validate and ensure p_nodes is a JSON array
  IF p_nodes IS NULL OR jsonb_typeof(p_nodes) != 'array' THEN
    v_nodes := '[]'::jsonb;
  ELSE
    v_nodes := p_nodes;
  END IF;

  -- Validate and ensure p_edges is a JSON array
  IF p_edges IS NULL OR jsonb_typeof(p_edges) != 'array' THEN
    v_edges := '[]'::jsonb;
  ELSE
    v_edges := p_edges;
  END IF;

  -- Update or insert workflow nodes if we have any
  IF jsonb_array_length(v_nodes) > 0 THEN
    WITH payload AS (
      SELECT * FROM jsonb_to_recordset(v_nodes)
      AS x(id uuid, type text, position jsonb, data jsonb)
    )
    INSERT INTO workflow_nodes AS wn
    (id, workflow_id, node_type, agent_id, position, data, tool_id, updated_at)
    SELECT 
      p.id,
      p_workflow_id,
      p.type,
      (p.data->'agent'->>'id')::uuid,
      coalesce(p.position, '{"x":0,"y":0,"flow":"main-flow"}'::jsonb),
      p.data,
      (p.data->>'toolId')::uuid,
      now()
    FROM payload p
    ON CONFLICT (id) DO UPDATE
    SET 
      node_type = excluded.node_type,
      agent_id = excluded.agent_id,
      position = excluded.position,
      data = excluded.data,
      tool_id = excluded.tool_id,
      updated_at = now();
    
    -- Get count of affected nodes
    GET DIAGNOSTICS v_node_count = ROW_COUNT;

    -- Delete nodes not in the payload
    DELETE FROM workflow_nodes wn
    WHERE wn.workflow_id = p_workflow_id
    AND wn.id NOT IN (
      SELECT x.id::uuid FROM jsonb_to_recordset(v_nodes) AS x(id uuid)
    );
  END IF;

  -- Update or insert workflow edges if we have any
  IF jsonb_array_length(v_edges) > 0 THEN
    WITH payload AS (
      SELECT * FROM jsonb_to_recordset(v_edges)
      AS x(id uuid, from_node_id uuid, to_node_id uuid, data jsonb)
    )
    INSERT INTO workflow_edges AS we 
    (id, workflow_id, from_node_id, to_node_id, trigger_intent, condition, updated_at)
    SELECT 
      p.id,
      p_workflow_id,
      p.from_node_id,
      p.to_node_id,
      -- Ensure trigger_intent is never null by providing a default value
      -- This fixes the issue with null values in trigger_intent column
      COALESCE(p.data->>'trigger_intent', p.data->>'label', 'default_intent'), 
      p.data->'condition',
      now()
    FROM payload p
    ON CONFLICT (id) DO UPDATE
    SET 
      from_node_id = excluded.from_node_id,
      to_node_id = excluded.to_node_id,
      trigger_intent = excluded.trigger_intent,
      condition = excluded.condition,
      updated_at = now();
    
    -- Get count of affected edges
    GET DIAGNOSTICS v_edge_count = ROW_COUNT;

    -- Delete edges not in the payload
    DELETE FROM workflow_edges we
    WHERE we.workflow_id = p_workflow_id
    AND we.id NOT IN (
      SELECT x.id::uuid FROM jsonb_to_recordset(v_edges) AS x(id uuid)
    );
  END IF;

  -- Update workflow's updated_at
  UPDATE workflows 
  SET 
    updated_at = now(),
    data = jsonb_set(
      COALESCE(data, '{}'::jsonb),
      '{stats}'::text[],
      jsonb_build_object(
        'node_count', v_node_count,
        'edge_count', v_edge_count,
        'last_synced', now()
      )::jsonb,
      true
    )
  WHERE id = p_workflow_id;

  -- Log the sync operation
  INSERT INTO logs (event_type, workflow_id, details)
  VALUES (
    'workflow_synced',
    p_workflow_id,
    jsonb_build_object(
      'nodes_processed', v_node_count,
      'edges_processed', v_edge_count,
      'timestamp', now()
    )
  );

EXCEPTION
  WHEN OTHERS THEN
    -- Log the error
    INSERT INTO logs (event_type, workflow_id, details, error)
    VALUES (
      'workflow_sync_error',
      p_workflow_id,
      jsonb_build_object(
        'nodes', p_nodes,
        'edges', p_edges,
        'timestamp', now()
      ),
      SQLERRM
    );
    -- Re-raise the exception
    RAISE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;