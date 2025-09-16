-- Seed common tools with idempotent upsert
INSERT INTO tools (id, name, description, function_code, parameters, is_active, created_at, updated_at)
VALUES 
(
  '550e8400-e29b-41d4-a716-446655440000',
  'http_request',
  'Make an HTTP request to a specified URL',
  'async def http_request(url: str, method: str = "GET", headers: dict = None, body: dict = None):
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, headers=headers, json=body)
        return {"status_code": response.status_code, "content": response.text}',
  '{"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]}, "headers": {"type": "object"}, "body": {"type": "object"}}, "required": ["url"]}',
  true,
  now(),
  now()
),
(
  '550e8400-e29b-41d4-a716-446655440001',
  'send_email',
  'Send an email using the configured SMTP server',
  'async def send_email(to: str, subject: str, body: str, from_email: str = null):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    msg = MIMEMultipart()
    msg["From"] = from_email or "noreply@agentflow.ai"
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP("smtp.example.com", 587) as server:
        server.starttls()
        server.login("user", "password")  # Should come from config
        server.send_message(msg)
    return {"status": "sent", "to": to, "subject": subject}',
  '{"type": "object", "properties": {"to": {"type": "string", "format": "email"}, "subject": {"type": "string"}, "body": {"type": "string"}, "from_email": {"type": "string", "format": "email"}}, "required": ["to", "subject", "body"]}',
  true,
  now(),
  now()
),
(
  '550e8400-e29b-41d4-a716-446655440002',
  'query_database',
  'Execute a SQL query against the database',
  'async def query_database(query: str, params: dict = None):
    from ..core.database import get_supabase_client
    
    supabase = get_supabase_client()
    try:
        result = await supabase.rpc("execute_sql", {"query": query, "params": params or {}})
        return {"status": "success", "data": result.data}
    except Exception as e:
        return {"status": "error", "error": str(e)}',
  '{"type": "object", "properties": {"query": {"type": "string"}, "params": {"type": "object"}}, "required": ["query"]}',
  true,
  now(),
  now()
)
ON CONFLICT (id) DO UPDATE 
SET 
  name = EXCLUDED.name,
  description = EXCLUDED.description,
  function_code = EXCLUDED.function_code,
  parameters = EXCLUDED.parameters,
  is_active = EXCLUDED.is_active,
  updated_at = now();

-- Create a function to log tool registration
CREATE OR REPLACE FUNCTION log_tool_registration()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    RAISE NOTICE 'ℹ️ Registered new tool: % (ID: %)', NEW.name, NEW.id;
  ELSIF TG_OP = 'UPDATE' THEN
    IF OLD.name != NEW.name OR OLD.function_code != NEW.function_code THEN
      RAISE NOTICE 'ℹ️ Updated tool: % (ID: %)', NEW.name, NEW.id;
    END IF;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for tool registration logging
DROP TRIGGER IF EXISTS tool_registration_trigger ON tools;
CREATE TRIGGER tool_registration_trigger
AFTER INSERT OR UPDATE ON tools
FOR EACH ROW EXECUTE FUNCTION log_tool_registration();

-- Create a view to see tool usage statistics
CREATE OR REPLACE VIEW tool_usage_stats AS
SELECT 
  t.id,
  t.name,
  t.description,
  COUNT(DISTINCT l.id) as execution_count,
  MAX(l.created_at) as last_used_at
FROM tools t
LEFT JOIN logs l ON l.details->>'tool_id' = t.id::text
WHERE t.is_active = true
GROUP BY t.id, t.name, t.description
ORDER BY execution_count DESC;
