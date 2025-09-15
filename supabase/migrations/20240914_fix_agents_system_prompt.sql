-- Add default value for system_prompt in agents table
ALTER TABLE IF EXISTS public.agents 
    ALTER COLUMN system_prompt SET NOT NULL,
    ALTER COLUMN system_prompt SET DEFAULT '';

-- Update any existing NULL values to empty string
UPDATE public.agents 
SET system_prompt = '' 
WHERE system_prompt IS NULL;
