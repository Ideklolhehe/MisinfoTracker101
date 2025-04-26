#!/usr/bin/env python
"""
Migration script to update the DetectedNarrative.title field from varchar(200) to text.
This script should be run as a one-time migration when the model is updated.
"""

import argparse
import sqlalchemy
from app import app, db
from sqlalchemy import create_engine, text
import logging

def migrate_title_field():
    """Migrate the title field in the detected_narrative table from varchar(200) to text."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting detected_narrative.title field migration")
    
    # Database URL from app config
    db_url = app.config.get('SQLALCHEMY_DATABASE_URI')
    
    if not db_url:
        logger.error("No database URL found in app configuration")
        return False
    
    try:
        # Create a direct database connection
        engine = create_engine(db_url)
        conn = engine.connect()
        
        # Check if the table exists
        inspector = sqlalchemy.inspect(engine)
        if 'detected_narrative' not in inspector.get_table_names():
            logger.error("detected_narrative table does not exist. Migration not needed.")
            conn.close()
            return False
        
        # Get current column type
        columns = inspector.get_columns('detected_narrative')
        title_col = None
        for col in columns:
            if col['name'] == 'title':
                title_col = col
                break
                
        if not title_col:
            logger.error("title column not found in detected_narrative table")
            conn.close()
            return False
            
        # Check if migration is needed
        if 'VARCHAR' not in str(title_col['type']).upper():
            logger.info("title column is not VARCHAR. Migration not needed.")
            conn.close()
            return True
            
        # Execute the migration
        logger.info("Altering title column type from VARCHAR to TEXT")
        if 'postgresql' in db_url:
            # PostgreSQL syntax
            conn.execute(text("ALTER TABLE detected_narrative ALTER COLUMN title TYPE text"))
        elif 'sqlite' in db_url:
            # SQLite cannot alter column types directly, so use a different approach
            # We create a new table, copy data, drop old table, rename new table
            logger.info("SQLite database detected, performing special migration")
            
            # Check if temporary table already exists
            if 'detected_narrative_new' in inspector.get_table_names():
                logger.info("Removing old temporary table from previous migration attempt")
                conn.execute(text("DROP TABLE detected_narrative_new"))
            
            # Create new table with TEXT column
            conn.execute(text("""
                CREATE TABLE detected_narrative_new (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    confidence_score FLOAT,
                    first_detected TIMESTAMP,
                    last_updated TIMESTAMP,
                    status VARCHAR(50),
                    language VARCHAR(10),
                    vector_id VARCHAR(100)
                )
            """))
            
            # Copy data
            conn.execute(text("""
                INSERT INTO detected_narrative_new 
                SELECT id, title, description, confidence_score, first_detected, 
                       last_updated, status, language, vector_id 
                FROM detected_narrative
            """))
            
            # Drop old table
            conn.execute(text("DROP TABLE detected_narrative"))
            
            # Rename new table
            conn.execute(text("ALTER TABLE detected_narrative_new RENAME TO detected_narrative"))
            
            logger.info("SQLite migration completed through table recreation")
        else:
            # Generic approach for other databases
            conn.execute(text("ALTER TABLE detected_narrative MODIFY COLUMN title TEXT NOT NULL"))
            
        conn.close()
        logger.info("Migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate DetectedNarrative.title field to TEXT type")
    parser.add_argument('--dry-run', action='store_true', help="Check migration status without modifying database")
    
    args = parser.parse_args()
    
    with app.app_context():
        if args.dry_run:
            # Just check if migration is needed
            print("Checking if migration is needed (dry run)")
            migrate_title_field()
        else:
            # Perform the migration
            result = migrate_title_field()
            if result:
                print("Migration completed successfully")
            else:
                print("Migration failed, check the logs for details")
