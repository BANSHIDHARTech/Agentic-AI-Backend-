-- Create workflow_nodes table
CREATE TABLE IF NOT EXISTS public.workflow_nodes (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    type TEXT NOT NULL,
    position JSONB NOT NULL DEFAULT '{"x": 0, "y": 0}'::jsonb,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    tool_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT,
    updated_by TEXT,
    
    CONSTRAINT fk_workflow
        FOREIGN KEY(workflow_id) 
        REFERENCES public.workflows(id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_tool
        FOREIGN KEY(tool_id)
        REFERENCES public.tools(id)
        ON DELETE SET NULL
);

-- Create workflow_edges table
CREATE TABLE IF NOT EXISTS public.workflow_edges (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    from_node_id TEXT NOT NULL,
    to_node_id TEXT NOT NULL,
    source_handle TEXT NOT NULL DEFAULT 'output',
    target_handle TEXT NOT NULL DEFAULT 'input',
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT,
    
    CONSTRAINT fk_workflow
        FOREIGN KEY(workflow_id) 
        REFERENCES public.workflows(id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_from_node
        FOREIGN KEY(from_node_id)
        REFERENCES public.workflow_nodes(id)
        ON DELETE CASCADE,
        
    CONSTRAINT fk_to_node
        FOREIGN KEY(to_node_id)
        REFERENCES public.workflow_nodes(id)
        ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_workflow_nodes_workflow_id ON public.workflow_nodes(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_edges_workflow_id ON public.workflow_edges(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_edges_from_node ON public.workflow_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_workflow_edges_to_node ON public.workflow_edges(to_node_id);

-- Create trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_workflow_nodes_updated_at
BEFORE UPDATE ON public.workflow_nodes
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_edges_updated_at
BEFORE UPDATE ON public.workflow_edges
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to validate node connections
CREATE OR REPLACE FUNCTION validate_node_connection()
RETURNS TRIGGER AS $$
DECLARE
    source_type TEXT;
    target_type TEXT;
    source_outputs JSONB;
    target_inputs JSONB;
BEGIN
    -- Get source node outputs
    SELECT data->'outputs' INTO source_outputs
    FROM public.workflow_nodes
    WHERE id = NEW.from_node_id;
    
    -- Get target node inputs
    SELECT data->'inputs' INTO target_inputs
    FROM public.workflow_nodes
    WHERE id = NEW.to_node_id;
    
    -- Check if source has the output handle
    IF NOT (source_outputs ? NEW.source_handle) THEN
        RAISE EXCEPTION 'Source node does not have output handle: %', NEW.source_handle;
    END IF;
    
    -- Check if target has the input handle
    IF NOT (target_inputs ? NEW.target_handle) THEN
        RAISE EXCEPTION 'Target node does not have input handle: %', NEW.target_handle;
    END IF;
    
    -- Get types for validation
    SELECT (source_outputs->>NEW.source_handle)::jsonb->>'type' INTO source_type;
    SELECT (target_inputs->>NEW.target_handle)::jsonb->>'type' INTO target_type;
    
    -- Validate type compatibility
    IF source_type IS DISTINCT FROM target_type THEN
        RAISE EXCEPTION 'Type mismatch: % != %', source_type, target_type;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for connection validation
CREATE TRIGGER validate_workflow_edge
BEFORE INSERT OR UPDATE ON public.workflow_edges
FOR EACH ROW EXECUTE FUNCTION validate_node_connection();

-- Create function to prevent circular dependencies
CREATE OR REPLACE FUNCTION prevent_circular_dependencies()
RETURNS TRIGGER AS $$
DECLARE
    cycle_found BOOLEAN;
BEGIN
    -- Check for cycles using a recursive CTE
    WITH RECURSIVE cycle_check AS (
        -- Start with the new edge
        SELECT NEW.from_node_id, NEW.to_node_id, 1 AS depth, ARRAY[NEW.from_node_id, NEW.to_node_id] AS path
        
        UNION ALL
        
        -- Follow edges in the same direction
        SELECT e.from_node_id, e.to_node_id, cc.depth + 1, cc.path || e.to_node_id
        FROM public.workflow_edges e
        JOIN cycle_check cc ON e.from_node_id = cc.to_node_id
        WHERE e.workflow_id = NEW.workflow_id
        AND e.to_node_id != ALL(cc.path)  -- Prevent infinite loops
        AND cc.depth < 100  -- Safety limit
    )
    -- Check if we've found a cycle back to the original source
    SELECT EXISTS (
        SELECT 1 FROM cycle_check 
        WHERE from_node_id = to_node_id
    ) INTO cycle_found;
    
    IF cycle_found THEN
        RAISE EXCEPTION 'Circular dependency detected';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to prevent circular dependencies
CREATE TRIGGER prevent_workflow_cycles
BEFORE INSERT OR UPDATE ON public.workflow_edges
FOR EACH ROW EXECUTE FUNCTION prevent_circular_dependencies();

-- Create function to update node data
CREATE OR REPLACE FUNCTION update_node_data(
    p_node_id TEXT,
    p_data JSONB
) RETURNS JSONB AS $$
DECLARE
    current_data JSONB;
    updated_data JSONB;
BEGIN
    -- Get current data
    SELECT data INTO current_data
    FROM public.workflow_nodes
    WHERE id = p_node_id
    FOR UPDATE;
    
    IF current_data IS NULL THEN
        RAISE EXCEPTION 'Node not found: %', p_node_id;
    END IF;
    
    -- Merge the new data with existing data
    updated_data := current_data || p_data;
    
    -- Update the node
    UPDATE public.workflow_nodes
    SET data = updated_data,
        updated_at = NOW()
    WHERE id = p_node_id;
    
    -- Log the update
    INSERT INTO logs (event_type, details)
    VALUES ('node_updated', jsonb_build_object(
        'node_id', p_node_id,
        'changes', p_data,
        'timestamp', NOW()
    ));
    
    RETURN updated_data;
END;
$$ LANGUAGE plpgsql;
