-- PostgreSQL initialization script for LegalQA with performance optimizations
-- This script sets up the database with optimal indices and configuration

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create optimized schema
DO $$
BEGIN
    -- Drop tables if they exist
    DROP TABLE IF EXISTS chunks CASCADE;
    DROP TABLE IF EXISTS documents CASCADE;
    
    -- Create documents table
    CREATE TABLE documents (
        doc_id VARCHAR PRIMARY KEY,
        source VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create chunks table with optimized structure
    CREATE TABLE chunks (
        chunk_id VARCHAR PRIMARY KEY,
        doc_id VARCHAR REFERENCES documents(doc_id) ON DELETE CASCADE,
        text TEXT NOT NULL,
        embedding vector(1536), -- OpenAI embedding dimension
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create performance indices
    RAISE NOTICE 'Creating performance indices...';
    
    -- Standard B-tree indices
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_created_at ON chunks(created_at);
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_source ON documents(source);
    
    -- Vector similarity indices (IVFFlat for better performance)
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_cosine 
    ON chunks USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);
    
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_l2 
    ON chunks USING ivfflat (embedding vector_l2_ops) 
    WITH (lists = 100);
    
    -- Partial indices for better performance
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_non_null_embedding 
    ON chunks(chunk_id) WHERE embedding IS NOT NULL;
    
    -- Text search index for full-text search capabilities
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_text_gin 
    ON chunks USING gin(to_tsvector('english', text));
    
    RAISE NOTICE 'Database schema and indices created successfully';
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error during initialization: %', SQLERRM;
END $$;

-- Optimize PostgreSQL settings for performance
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '4MB';

-- Enable query logging for performance monitoring
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_duration = 'on';
ALTER SYSTEM SET log_min_duration_statement = 1000;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at 
    BEFORE UPDATE ON chunks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Analyze tables for better query planning
ANALYZE documents;
ANALYZE chunks;

-- Create a view for monitoring database performance
CREATE OR REPLACE VIEW performance_stats AS
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables 
WHERE tablename IN ('documents', 'chunks');

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${POSTGRES_USER};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${POSTGRES_USER};

-- Final message
DO $$
BEGIN
    RAISE NOTICE '✅ LegalQA database initialization completed successfully!';
    RAISE NOTICE 'Tables created: documents, chunks';
    RAISE NOTICE 'Indices created: B-tree, IVFFlat vector, GIN text search';
    RAISE NOTICE 'Performance optimizations applied';
END $$;