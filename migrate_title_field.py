"""
Migration script to update the DetectedNarrative.title field from varchar(200) to text.
This script should be run as a one-time migration when the model is updated.
"""

import sys
import logging
from app import app, db
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_title_field():
    """Migrate the title field in the detected_narrative table from varchar(200) to text."""
    try:
        # Use SQLAlchemy core to execute ALTER TABLE command
        with db.engine.connect() as conn:
            # We need to use raw SQL for this migration as SQLAlchemy ORM doesn't handle
            # type changes during normal operations
            conn.execute(text("ALTER TABLE detected_narrative ALTER COLUMN title TYPE TEXT"))
            conn.commit()
            
        logger.info("Migration successful: Changed detected_narrative.title from varchar(200) to text")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    with app.app_context():
        # Run the migration
        if migrate_title_field():
            logger.info("Migration completed successfully")
            sys.exit(0)
        else:
            logger.error("Migration failed")
            sys.exit(1)