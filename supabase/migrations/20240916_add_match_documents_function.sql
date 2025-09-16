-- Create match_documents function that's compatible with the KnowledgeService implementation
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.1,
  match_count integer DEFAULT 10,
  session_id text DEFAULT NULL,
  user_id text DEFAULT NULL
) 
RETURNS TABLE(
  id uuid,
  content text,
  metadata jsonb,
  source_type text,
  source_reference text,
  chunk_index integer,
  total_chunks integer,
  similarity float
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    d.id,
    d.content,
    d.metadata,
    d.source_type,
    d.source_reference,
    d.chunk_index,
    d.total_chunks,
    (1 - (d.embedding <=> query_embedding))::float as similarity
  FROM documents d
  WHERE d.embedding IS NOT NULL
    AND (1 - (d.embedding <=> query_embedding)) > match_threshold
    AND (session_id IS NULL OR d.session_id = session_id)
    AND (user_id IS NULL OR d.user_id = user_id)
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- First drop the existing function if it exists
DROP FUNCTION IF EXISTS get_knowledge_sessions(int, int, text);

-- Create get_knowledge_sessions function that's required by the sessions endpoint
CREATE OR REPLACE FUNCTION get_knowledge_sessions(
  p_limit int DEFAULT 100,
  p_offset int DEFAULT 0,
  p_user_id text DEFAULT NULL
)
RETURNS jsonb AS $$
DECLARE
  v_result jsonb;
  v_count integer;
  v_data jsonb;
BEGIN
  -- Get the count of total sessions
  SELECT COUNT(DISTINCT session_id) INTO v_count
  FROM documents d
  WHERE (p_user_id IS NULL OR d.user_id = p_user_id);
  
  -- Get the session data
  SELECT jsonb_agg(row_to_json(t)) INTO v_data
  FROM (
    SELECT
      d.session_id,
      d.user_id,
      COUNT(*) as document_count,
      MIN(d.created_at) as first_created_at,
      MAX(d.created_at) as last_updated_at,
      ARRAY_AGG(DISTINCT d.source_type) as source_types
    FROM
      documents d
    WHERE
      (p_user_id IS NULL OR d.user_id = p_user_id)
    GROUP BY
      d.session_id, d.user_id
    ORDER BY
      MAX(d.created_at) DESC
    LIMIT p_limit
    OFFSET p_offset
  ) t;
  
  -- Handle NULL case
  IF v_data IS NULL THEN
    v_data := '[]'::jsonb;
  END IF;
  
  -- Construct the result
  v_result := jsonb_build_object(
    'data', v_data,
    'count', v_count
  );
  
  RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add helpful comments to explain the function purposes
COMMENT ON FUNCTION match_documents IS 'Performs vector similarity search on documents with additional filtering options';
COMMENT ON FUNCTION get_knowledge_sessions IS 'Lists all knowledge sessions with summary statistics';

-- Insert a log entry for this migration
INSERT INTO logs (event_type, details) 
VALUES (
  'knowledge_base_migration',
  jsonb_build_object(
    'message', 'Added match_documents and get_sessions functions for knowledge API compatibility',
    'timestamp', now(),
    'migration', '20240916_add_match_documents_function'
  )
);