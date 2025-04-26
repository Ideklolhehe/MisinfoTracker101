#!/usr/bin/env python
"""
Migration script to add meta_data field to the DetectedNarrative model.
"""

import logging
import sqlalchemy
from app import app, db
from sqlalchemy import create_engine, text

def add_metadata_field():
    """Add meta_data field to the detected_narrative table."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting migration to add meta_data field to detected_narrative table")
    
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
            logger.error("detected_narrative table does not exist.")
            conn.close()
            return False
        
        # Check if meta_data column already exists
        columns = inspector.get_columns('detected_narrative')
        meta_data_exists = False
        for col in columns:
            if col['name'] == 'meta_data':
                meta_data_exists = True
                break
                
        if meta_data_exists:
            logger.info("meta_data column already exists. No migration needed.")
            conn.close()
            return True
        
        # Add the meta_data column to the table
        logger.info("Adding meta_data column to detected_narrative table")
        if 'postgresql' in db_url:
            # PostgreSQL syntax
            conn.execute(text("ALTER TABLE detected_narrative ADD COLUMN meta_data TEXT"))
        elif 'sqlite' in db_url:
            # SQLite syntax
            conn.execute(text("ALTER TABLE detected_narrative ADD COLUMN meta_data TEXT"))
        else:
            # Generic SQL syntax
            conn.execute(text("ALTER TABLE detected_narrative ADD COLUMN meta_data TEXT"))
            
        conn.close()
        logger.info("Migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    print("Adding meta_data field to DetectedNarrative model...")
    with app.app_context():
        if add_metadata_field():
            print("Migration completed successfully")
        else:
            print("Migration failed, check the logs for details")
