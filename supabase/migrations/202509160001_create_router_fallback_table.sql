-- Create the router_fallback_messages table if it doesn't exist

-- Check if the table already exists before creating it
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename = 'router_fallback_messages'
    ) THEN
        -- Create the table
        CREATE TABLE public.router_fallback_messages (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            message TEXT NOT NULL,
            description TEXT,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        );

        -- Add some default fallback messages
        INSERT INTO public.router_fallback_messages (message, description, is_active)
        VALUES 
            ('I''m not sure how to help with that. Could you try rephrasing your question?', 'General fallback message', true),
            ('I don''t have enough information to answer that. Could you provide more details?', 'Information request fallback', true),
            ('I''m still learning and don''t know how to respond to that yet.', 'Learning phase fallback', true);

        -- Create an updated_at trigger
        CREATE TRIGGER set_timestamp
        BEFORE UPDATE ON public.router_fallback_messages
        FOR EACH ROW
        EXECUTE PROCEDURE update_timestamp_column();
    END IF;
END $$;