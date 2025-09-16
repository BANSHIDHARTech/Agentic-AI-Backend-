-- Add error column to logs table
ALTER TABLE logs ADD COLUMN IF NOT EXISTS error text;