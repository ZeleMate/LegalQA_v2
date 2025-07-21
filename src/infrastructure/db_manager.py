import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Optimized database manager with connection pooling and async support."""
    
    def __init__(self):
        self.pool = None
        self.sync_pool = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize async database connection pool."""
        if self._initialized:
            return
        
        load_dotenv()
        
        # Database configuration
        db_config = {
            'host': os.getenv("POSTGRES_HOST", "localhost"),
            'port': int(os.getenv("POSTGRES_PORT", 5432)),
            'database': os.getenv("POSTGRES_DB"),
            'user': os.getenv("POSTGRES_USER"),
            'password': os.getenv("POSTGRES_PASSWORD"),
        }
        
        try:
            # Try to use asyncpg for better performance
            import asyncpg
            self.pool = await asyncpg.create_pool(
                **db_config,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Async database pool initialized with asyncpg")
            
        except ImportError:
            logger.warning("asyncpg not available, falling back to synchronous connections")
            # Fallback to psycopg2 with connection pooling
            try:
                import psycopg2.pool
                connection_string = (
                    f"host={db_config['host']} "
                    f"port={db_config['port']} "
                    f"dbname={db_config['database']} "
                    f"user={db_config['user']} "
                    f"password={db_config['password']}"
                )
                self.sync_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=5,
                    maxconn=20,
                    dsn=connection_string
                )
                logger.info("Sync database pool initialized with psycopg2")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise
        
        self._initialized = True
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        if self.pool:
            # Async connection with better error handling
            connection = None
            try:
                connection = await self.pool.acquire()
                yield connection
            except Exception as e:
                logger.error(f"Error acquiring connection: {e}")
                raise
            finally:
                if connection:
                    try:
                        await self.pool.release(connection)
                    except Exception as e:
                        logger.warning(f"Error releasing connection: {e}")
        elif self.sync_pool:
            # Sync connection in thread pool
            connection = None
            try:
                connection = self.sync_pool.getconn()
                yield connection
            finally:
                if connection:
                    self.sync_pool.putconn(connection)
        else:
            raise RuntimeError("No database pool available")
    
    async def fetch_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Optimized method to fetch document chunks by IDs.
        Uses batch queries and optimized data processing.
        """
        if not chunk_ids:
            return {}

        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(chunk_ids))

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.get_connection() as conn:
                    if self.pool:  # asyncpg
                        query = """
                            SELECT chunk_id, doc_id, text, embedding
                            FROM chunks 
                            WHERE chunk_id = ANY($1::text[])
                        """
                        rows = await conn.fetch(query, unique_ids)

                        docs_data = {}
                        for row in rows:
                            chunk_id = row['chunk_id']
                            docs_data[chunk_id] = {
                                'chunk_id': chunk_id,
                                'doc_id': row['doc_id'],
                                'text': row['text'],
                                'embedding': self._parse_pgvector_embedding(row['embedding'])
                            }
                    else:  # psycopg2
                        with conn.cursor() as cursor:
                            query = """
                                SELECT chunk_id, doc_id, text, embedding
                                FROM chunks 
                                WHERE chunk_id = ANY(%s)
                            """
                            cursor.execute(query, (unique_ids,))
                            rows = cursor.fetchall()

                            # Get column names
                            colnames = [desc[0] for desc in cursor.description]

                            docs_data = {}
                            for row in rows:
                                row_dict = dict(zip(colnames, row))
                                chunk_id = row_dict['chunk_id']
                                row_dict['embedding'] = self._parse_pgvector_embedding(row_dict['embedding'])
                                docs_data[chunk_id] = row_dict

                logger.debug(f"Fetched {len(docs_data)} chunks from database")
                return docs_data

            except Exception as e:
                logger.warning(f"Database fetch attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All database fetch attempts failed for chunk_ids: {chunk_ids[:5]}...")
                    return {}
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
    
    def _parse_pgvector_embedding(self, embedding_str: str) -> Optional[str]:
        """
        Parse pgvector embedding string back to numpy array.
        Optimized version with better error handling.
        """
        if not embedding_str:
            return None
        
        try:
            # pgvector returns format like "[1.0,2.0,3.0]"
            # Remove brackets and split by comma
            clean_str = embedding_str.strip('[]')
            if not clean_str:
                return None
            
            # Convert to numpy array more efficiently
            values = np.fromstring(clean_str, sep=',', dtype=np.float32)
            return values.tobytes().hex()  # Return as hex string for consistency
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse embedding: {e}")
            return None
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a general query and return results."""
        async with self.get_connection() as conn:
            if self.pool:  # asyncpg
                if params:
                    rows = await conn.fetch(query, *params)
                else:
                    rows = await conn.fetch(query)
                return [dict(row) for row in rows]
            else:  # psycopg2
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    colnames = [desc[0] for desc in cursor.description]
                    return [dict(zip(colnames, row)) for row in rows]
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        stats_query = """
            SELECT 
                schemaname,
                relname as tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_rows,
                n_dead_tup as dead_rows
            FROM pg_stat_user_tables 
            WHERE relname IN ('documents', 'chunks')
        """
        
        try:
            results = await self.execute_query(stats_query)
            return {row['tablename']: row for row in results}
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def optimize_database(self):
        """Run database optimization queries."""
        optimization_queries = [
            # Analyze tables for better query planning
            "ANALYZE chunks;",
            "ANALYZE documents;",
            
            # Create indices if they don't exist (only if not already created)
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_chunk_id 
            ON chunks(chunk_id);
            """,
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_doc_id 
            ON chunks(doc_id);
            """,
        ]
        
        for query in optimization_queries:
            try:
                async with self.get_connection() as conn:
                    if self.pool:
                        await conn.execute(query)
                    else:
                        with conn.cursor() as cursor:
                            cursor.execute(query)
                            conn.commit()
                logger.info(f"Executed optimization query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Optimization query failed: {e}")
                # Don't fail the entire optimization if one query fails
                continue
    
    async def close(self):
        """Closes the database connection pool."""
        if self.pool:
            await self.pool.close()
        if self.sync_pool:
            self.sync_pool.closeall()
        self._initialized = False
        logger.info("Database connection pool closed.")


# Singleton instance of the database manager
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Returns a singleton instance of the DatabaseManager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def fetch_chunks(chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Function to fetch chunks with caching support."""
    db_manager = get_db_manager()
    return await db_manager.fetch_chunks_by_ids(chunk_ids)


async def ensure_database_setup():
    """Ensure database is set up with proper indices."""
    db_manager = get_db_manager()
    await db_manager.optimize_database()


# Context manager for database operations
@asynccontextmanager
async def database_session():
    """Context manager for database operations."""
    db = get_db_manager()
    await db.initialize()
    try:
        yield db
    finally:
        # Connection cleanup is handled by the pool
        pass