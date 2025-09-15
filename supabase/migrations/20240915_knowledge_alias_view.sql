/*
  Create knowledge_chunks view as an alias to the documents table
  
  This view provides backwards compatibility with code that may reference
  the 'knowledge_chunks' table which has been renamed to 'documents'
*/

-- Create a view named knowledge_chunks that points to the documents table
CREATE OR REPLACE VIEW knowledge_chunks AS 
SELECT * FROM documents;

-- Create a comment explaining the view's purpose
COMMENT ON VIEW knowledge_chunks IS 'Compatibility view that maps to documents table for legacy code support';

-- Add to logs
INSERT INTO logs (event_type, details) 
VALUES (
  'knowledge_compatibility_view_created', 
  jsonb_build_object(
    'message', 'Created knowledge_chunks view as compatibility alias to documents table',
    'timestamp', now()
  )
);

-- Success message
DO $$
BEGIN
  RAISE NOTICE '=== Knowledge Chunks Compatibility View Created ===';
  RAISE NOTICE 'Created view knowledge_chunks -> documents';
  RAISE NOTICE 'This view provides compatibility for code using the older table name';
  RAISE NOTICE '===================================================';
END $$;