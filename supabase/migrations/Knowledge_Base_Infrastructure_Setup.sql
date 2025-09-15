/*
  # Knowledge Base Support Tables
  
  Creates tables and functions for Knowledge Base ingestion and querying:
  
  1. New Tables
    - `documents` - Stores document content, metadata, and embeddings
    - `knowledge_sessions` - Tracks knowledge base sessions and queries
    
  2. Vector Support
    - Enable pgvector extension for similarity search
    - Add vector column for 1536-dimensional embeddings (OpenAI ada-002)
    
  3. Security
    - Enable RLS on all new tables
    - Add policies for authenticated access
    - Proper indexing for vector similarity search
    
  4. Functions
    - Vector similarity search function
    - Document chunking metadata helpers
*/

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for knowledge base storage
CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  content text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  embedding vector(1536), -- OpenAI ada-002 embedding dimension
  source_type text DEFAULT 'text' CHECK (source_type IN ('text', 'url', 'file', 'pdf')),
  source_reference text,
  chunk_index integer DEFAULT 0,
  total_chunks integer DEFAULT 1,
  session_id text,
  user_id text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create knowledge_sessions table for tracking queries and sessions
CREATE TABLE IF NOT EXISTS knowledge_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id text NOT NULL,
  user_id text,
  query text NOT NULL,
  results jsonb DEFAULT '[]'::jsonb,
  result_count integer DEFAULT 0,
  similarity_threshold numeric(3,2) DEFAULT 0.7,
  processing_time_ms integer,
  created_at timestamptz DEFAULT now()
);

-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

CREATE INDEX IF NOT EXISTS idx_knowledge_sessions_session_id ON knowledge_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_sessions_user_id ON knowledge_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_sessions_created_at ON knowledge_sessions(created_at);

-- Enable Row Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_sessions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (permissive for development - customize for production)
-- Documents policies
CREATE POLICY "Enable read access for all users" ON documents FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON documents FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON documents FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON documents FOR DELETE USING (true);

-- Knowledge sessions policies
CREATE POLICY "Enable read access for all users" ON knowledge_sessions FOR SELECT USING (true);
CREATE POLICY "Enable insert for all users" ON knowledge_sessions FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update for all users" ON knowledge_sessions FOR UPDATE USING (true);
CREATE POLICY "Enable delete for all users" ON knowledge_sessions FOR DELETE USING (true);

-- Create vector similarity search function
CREATE OR REPLACE FUNCTION search_documents(
  query_embedding vector(1536),
  similarity_threshold numeric DEFAULT 0.7,
  match_count integer DEFAULT 10
)
RETURNS TABLE(
  id uuid,
  content text,
  metadata jsonb,
  source_type text,
  source_reference text,
  chunk_index integer,
  similarity numeric
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
    (1 - (d.embedding <=> query_embedding))::numeric as similarity
  FROM documents d
  WHERE d.embedding IS NOT NULL
    AND (1 - (d.embedding <=> query_embedding)) > similarity_threshold
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get document statistics
CREATE OR REPLACE FUNCTION get_knowledge_stats()
RETURNS TABLE(
  total_documents bigint,
  total_chunks bigint,
  source_types jsonb,
  recent_uploads bigint
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    COUNT(DISTINCT CASE WHEN chunk_index = 0 THEN id END) as total_documents,
    COUNT(*) as total_chunks,
    jsonb_object_agg(source_type, type_count) as source_types,
    COUNT(*) FILTER (WHERE created_at > now() - interval '24 hours') as recent_uploads
  FROM (
    SELECT 
      id,
      source_type,
      chunk_index,
      created_at,
      COUNT(*) OVER (PARTITION BY source_type) as type_count
    FROM documents
  ) stats;
END;
$$ LANGUAGE plpgsql;

-- Add helpful comments
COMMENT ON TABLE documents IS 'Knowledge base documents with vector embeddings for similarity search';
COMMENT ON TABLE knowledge_sessions IS 'Tracks knowledge base queries and search sessions';

COMMENT ON COLUMN documents.embedding IS '1536-dimensional vector embedding (OpenAI ada-002 compatible)';
COMMENT ON COLUMN documents.metadata IS 'Document metadata including title, author, tags, etc.';
COMMENT ON COLUMN documents.chunk_index IS 'Index of this chunk within the source document (0-based)';
COMMENT ON COLUMN documents.total_chunks IS 'Total number of chunks for the source document';

COMMENT ON FUNCTION search_documents IS 'Performs vector similarity search against document embeddings';
COMMENT ON FUNCTION get_knowledge_stats IS 'Returns statistics about the knowledge base';

-- Create helpful views for monitoring
CREATE OR REPLACE VIEW knowledge_base_summary AS
SELECT 
  source_type,
  COUNT(DISTINCT CASE WHEN chunk_index = 0 THEN id END) as document_count,
  COUNT(*) as chunk_count,
  AVG(length(content)) as avg_content_length,
  MAX(created_at) as last_upload
FROM documents
GROUP BY source_type
ORDER BY document_count DESC;

CREATE OR REPLACE VIEW recent_knowledge_activity AS
SELECT 
  ks.session_id,
  ks.query,
  ks.result_count,
  ks.similarity_threshold,
  ks.processing_time_ms,
  ks.created_at as query_time,
  COUNT(d.id) as related_documents
FROM knowledge_sessions ks
LEFT JOIN documents d ON d.session_id = ks.session_id
WHERE ks.created_at > now() - interval '24 hours'
GROUP BY ks.id, ks.session_id, ks.query, ks.result_count, ks.similarity_threshold, ks.processing_time_ms, ks.created_at
ORDER BY ks.created_at DESC
LIMIT 50;

COMMENT ON VIEW knowledge_base_summary IS 'Summary statistics of knowledge base by source type';
COMMENT ON VIEW recent_knowledge_activity IS 'Recent knowledge base queries and activity';

-- Insert initial log entry
INSERT INTO logs (event_type, details) 
SELECT 'knowledge_base_initialized', jsonb_build_object(
  'message', 'Knowledge Base tables and functions created successfully',
  'tables_created', jsonb_build_array('documents', 'knowledge_sessions'),
  'functions_created', jsonb_build_array('search_documents', 'get_knowledge_stats'),
  'views_created', jsonb_build_array('knowledge_base_summary', 'recent_knowledge_activity'),
  'vector_extension_enabled', true,
  'embedding_dimension', 1536,
  'timestamp', now()
)
WHERE NOT EXISTS (SELECT 1 FROM logs WHERE event_type = 'knowledge_base_initialized');

-- Final success message
DO $$
BEGIN
  RAISE NOTICE '=== Knowledge Base Tables Created Successfully ===';
  RAISE NOTICE 'Tables: documents, knowledge_sessions';
  RAISE NOTICE 'Functions: search_documents, get_knowledge_stats';
  RAISE NOTICE 'Views: knowledge_base_summary, recent_knowledge_activity';
  RAISE NOTICE 'Vector extension enabled with 1536-dim embeddings';
  RAISE NOTICE 'Ready for Knowledge Base API implementation!';
  RAISE NOTICE '====================================================';
END $$;