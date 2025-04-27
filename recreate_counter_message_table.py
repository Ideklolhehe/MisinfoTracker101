#!/usr/bin/env python3
"""
Script to drop and recreate the counter_message table
to update the strategy field from varchar(100) to text.
"""

import sys
import logging
import os
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CIVILIAN')

def recreate_counter_message_table():
    """Drop and recreate the counter_message table to update field definitions."""
    try:
        # Import the app and db within the function to avoid app context errors
        from app import app, db
        from models import CounterMessage
        
        with app.app_context():
            logger.info("Starting recreation of counter_message table")
            
            # First, drop the table
            conn = db.engine.connect()
            try:
                conn.execute(text("DROP TABLE IF EXISTS counter_message"))
                logger.info("Dropped counter_message table")
            except Exception as e:
                logger.error(f"Error dropping counter_message table: {e}")
                raise
            
            # Now recreate the table with the new schema from SQLAlchemy models
            db.create_all()
            logger.info("Recreated counter_message table with new schema")
            
            conn.close()
            logger.info("Table recreation completed successfully")
    
    except Exception as e:
        logger.error(f"Table recreation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting counter_message table recreation script")
    recreate_counter_message_table()
    logger.info("Table recreation script completed")