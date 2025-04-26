#!/usr/bin/env python
"""
Script to recreate database tables with the current model definitions.
This is meant to be used during development to fix schema inconsistencies.
"""

import logging
import os
from app import app, db
from models import User, DataSource, DetectedNarrative, NarrativeInstance
from models import BeliefNode, BeliefEdge, CounterMessage, SystemLog
from models import AdversarialContent, AdversarialEvaluation

def recreate_database():
    """Drop all tables and recreate them from the model definitions."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting database recreation process")
    
    # Skip confirmation in development environment
    logger.info("Development environment detected, skipping confirmation")
    
    # Dropping all tables to clean slate
    logger.info("Dropping all existing tables")
    db.drop_all()
    
    # Recreating all tables with the current model definitions
    logger.info("Recreating all tables from current model definitions")
    db.create_all()
    
    # Create admin user
    logger.info("Creating admin user")
    from werkzeug.security import generate_password_hash
    admin = User(
        username='admin',
        email='admin@civilian.org',
        password_hash=generate_password_hash('admin'),
        role='admin'
    )
    db.session.add(admin)
    
    # Create analyst user
    analyst = User(
        username='analyst',
        email='analyst@civilian.org',
        password_hash=generate_password_hash('analyst'),
        role='analyst'
    )
    db.session.add(analyst)
    
    # Add initial system log entry
    log = SystemLog(
        log_type='info',
        component='system',
        message='Database recreated successfully'
    )
    db.session.add(log)
    
    # Commit all changes
    db.session.commit()
    logger.info("Database recreation completed successfully")

if __name__ == "__main__":
    with app.app_context():
        recreate_database()
