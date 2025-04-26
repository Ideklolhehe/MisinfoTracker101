#!/usr/bin/env python
"""
Script to verify and standardize model fields and methods across the CIVILIAN system.
This script ensures all models with meta_data fields have proper accessor methods.
"""

import logging
import inspect
from app import app, db
from sqlalchemy.inspection import inspect as sqla_inspect
from sqlalchemy import Column, Text

def verify_models():
    """Verify and report on model standardization."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model verification")
    
    # Find all SQLAlchemy models
    from models import (User, DataSource, DetectedNarrative, NarrativeInstance,
                       BeliefNode, BeliefEdge, CounterMessage, SystemLog,
                       AdversarialContent, AdversarialEvaluation)
    
    models = [
        User, DataSource, DetectedNarrative, NarrativeInstance,
        BeliefNode, BeliefEdge, CounterMessage, SystemLog,
        AdversarialContent, AdversarialEvaluation
    ]
    
    # Verify each model
    needs_fix = []
    for model in models:
        model_name = model.__name__
        logger.info(f"Checking model: {model_name}")
        
        # Check if model has a meta_data field
        has_meta_data = False
        for column in sqla_inspect(model).columns:
            if column.name == 'meta_data' and isinstance(column.type, Text):
                has_meta_data = True
                break
        
        # Check if model has set_meta_data and get_meta_data methods
        has_set_method = hasattr(model, 'set_meta_data') and callable(getattr(model, 'set_meta_data'))
        has_get_method = hasattr(model, 'get_meta_data') and callable(getattr(model, 'get_meta_data'))
        
        # Report findings
        if has_meta_data:
            logger.info(f"  - Has meta_data field: Yes")
            if has_set_method and has_get_method:
                logger.info(f"  - Has meta_data methods: Yes")
            else:
                logger.info(f"  - Has meta_data methods: No")
                needs_fix.append((model_name, has_meta_data, has_set_method, has_get_method))
        else:
            logger.info(f"  - Has meta_data field: No")
            if has_set_method or has_get_method:
                logger.info(f"  - Has meta_data methods: Inconsistent")
                needs_fix.append((model_name, has_meta_data, has_set_method, has_get_method))
            else:
                logger.info(f"  - Has meta_data methods: N/A")
    
    # Report models that need fixes
    if needs_fix:
        logger.info("\nModels that need standardization:")
        for model_name, has_field, has_set, has_get in needs_fix:
            issues = []
            if has_field and not (has_set and has_get):
                issues.append("Missing methods for meta_data field")
            elif not has_field and (has_set or has_get):
                issues.append("Has methods but no meta_data field")
            logger.info(f"  - {model_name}: {', '.join(issues)}")
    else:
        logger.info("\nAll models are properly standardized!")

if __name__ == "__main__":
    with app.app_context():
        verify_models()
