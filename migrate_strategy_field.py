#!/usr/bin/env python3
"""
Migration script to update the CounterMessage.strategy field from varchar(100) to text.
This script should be run as a one-time migration when the model is updated.
"""

import sys
import logging
import os
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CIVILIAN')

def migrate_strategy_field():
    """Migrate the strategy field in the counter_message table from varchar(100) to text."""
    try:
        # Import the app and db within the function to avoid app context errors
        from app import app, db
        
        with app.app_context():
            logger.info("Starting migration of 'strategy' field in counter_message table")
            
            # Check if the column is already TEXT type
            conn = db.engine.connect()
            
            # Get the current datatype
            result = conn.execute(text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name='counter_message' AND column_name='strategy'"
            ))
            
            current_type = result.scalar()
            if current_type and current_type.lower() == 'text':
                logger.info("strategy field is already of type TEXT, no migration needed")
                return
            
            # Alter the column type to TEXT
            try:
                conn.execute(text("ALTER TABLE counter_message ALTER COLUMN strategy TYPE text"))
                logger.info("Successfully altered strategy column type to TEXT")
            except Exception as e:
                logger.error(f"Error altering strategy column: {e}")
                raise
            
            conn.close()
            logger.info("Migration completed successfully")
    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting strategy field migration script")
    migrate_strategy_field()
    logger.info("Migration script completed")