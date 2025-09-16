-- Migration to update knowledge schema and fix documents table
-- This migration adds any missing columns or updates existing tables

-- First, check if the documents table exists
-- If not, we need to create it
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'documents') THEN
        CREATE TABLE public.documents (
            id uuid PRIMARY KEY,
            session_id VARCHAR NOT NULL,
            user_id VARCHAR DEFAULT 'anonymous',
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            embedding vector(1536),
            source_type VARCHAR,
            source_reference VARCHAR,
            chunk_index INTEGER DEFAULT 0,
            total_chunks INTEGER DEFAULT 1,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Create an index on session_id for faster queries
        CREATE INDEX idx_documents_session_id ON public.documents(session_id);
        
        -- Create an index on user_id
        CREATE INDEX idx_documents_user_id ON public.documents(user_id);

        -- Create a gin index on metadata for JSON querying
        CREATE INDEX idx_documents_metadata ON public.documents USING gin(metadata);

        -- Add comment
        COMMENT ON TABLE public.documents IS 'Stores knowledge base documents with metadata and embeddings';
    END IF;
END $$;

-- Now add any missing columns to documents table 
DO $$ 
BEGIN
    -- Add embedding column if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_attribute WHERE attrelid = 'documents'::regclass AND attname = 'embedding') THEN
        ALTER TABLE public.documents ADD COLUMN embedding vector(1536);
    END IF;

    -- Add metadata column if it doesn't exist 
    IF NOT EXISTS (SELECT FROM pg_attribute WHERE attrelid = 'documents'::regclass AND attname = 'metadata') THEN
        ALTER TABLE public.documents ADD COLUMN metadata JSONB DEFAULT '{}'::jsonb;
    END IF;

    -- Add source_type column if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_attribute WHERE attrelid = 'documents'::regclass AND attname = 'source_type') THEN
        ALTER TABLE public.documents ADD COLUMN source_type VARCHAR;
    END IF;

    -- Add source_reference column if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_attribute WHERE attrelid = 'documents'::regclass AND attname = 'source_reference') THEN
        ALTER TABLE public.documents ADD COLUMN source_reference VARCHAR;
    END IF;
    
    -- Add chunk_index column if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_attribute WHERE attrelid = 'documents'::regclass AND attname = 'chunk_index') THEN
        ALTER TABLE public.documents ADD COLUMN chunk_index INTEGER DEFAULT 0;
    END IF;
    
    -- Add total_chunks column if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_attribute WHERE attrelid = 'documents'::regclass AND attname = 'total_chunks') THEN
        ALTER TABLE public.documents ADD COLUMN total_chunks INTEGER DEFAULT 1;
    END IF;
END $$;

-- Now create a function to insert documents and handle errors gracefully
CREATE OR REPLACE FUNCTION insert_document(
    doc_id UUID, 
    session_id TEXT, 
    content TEXT,
    user_id TEXT DEFAULT 'anonymous',
    metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    -- Insert the document with minimal fields
    INSERT INTO public.documents(id, session_id, user_id, content, metadata, created_at)
    VALUES (doc_id, session_id, user_id, content, metadata, now())
    RETURNING jsonb_build_object(
        'id', id,
        'success', true
    ) INTO result;
    
    RETURN result;
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object(
        'error', SQLERRM,
        'success', false
    );
END;
$$ LANGUAGE plpgsql;

-- Create a stored procedure for retrieving document information by session
CREATE OR REPLACE FUNCTION get_session_documents(
    p_session_id TEXT
) RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        CASE WHEN d.metadata IS NULL THEN '{}'::jsonb ELSE d.metadata END as metadata,
        d.created_at
    FROM 
        public.documents d
    WHERE 
        d.session_id = p_session_id
    ORDER BY
        d.created_at DESC;
END;
$$ LANGUAGE plpgsql;