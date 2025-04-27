"""
Script to temporarily fix authentication issues in the CIVILIAN system.
This should be used during development to disable login requirements.
"""

import os
import re
import importlib
import logging

logger = logging.getLogger(__name__)

def patch_replit_auth():
    """
    Apply a monkey patch to the Replit Auth module to bypass authentication for development.
    """
    try:
        # Import the replit_auth module
        import replit_auth
        from functools import wraps
        
        # Create a replacement for the require_login decorator
        def dev_login_decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Just call the function directly without any auth checks
                return f(*args, **kwargs)
            return decorated_function
        
        # Replace the require_login decorator
        replit_auth.require_login = dev_login_decorator
        
        # Add a mock replit token
        import werkzeug.local
        from flask import g
        
        class MockReplit:
            def __init__(self):
                self.token = {"expires_in": 3600}
        
        # Monkey patch the replit proxy
        old_replit = replit_auth.replit
        if isinstance(old_replit, werkzeug.local.LocalProxy):
            replit_auth.replit = werkzeug.local.LocalProxy(lambda: MockReplit())
        
        logger.warning("Applied monkey patch to replit_auth for development")
        return True
    except Exception as e:
        logger.error(f"Failed to patch replit_auth: {e}")
        return False

def update_route_files():
    """Update route files to import flask_login properly."""
    try:
        # Add missing flask_login import to any route files that need it
        routes_dir = os.path.join(os.path.dirname(__file__), 'routes')
        for filename in os.listdir(routes_dir):
            if filename.endswith('.py'):
                filepath = os.path.join(routes_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Skip if it already has the import
                if 'from flask_login import current_user' in content:
                    continue
                
                # Add the import if it's missing
                if 'current_user' in content and 'from flask_login import' not in content:
                    logger.info(f"Adding flask_login import to {filename}")
                    new_content = re.sub(
                        r'from flask import (.*)',
                        r'from flask import \1\nfrom flask_login import current_user',
                        content
                    )
                    with open(filepath, 'w') as f:
                        f.write(new_content)
        
        return True
    except Exception as e:
        logger.error(f"Failed to update route files: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Applying authentication fixes for development...")
    
    # Apply Replit Auth patch
    if patch_replit_auth():
        print("✓ Successfully patched Replit Auth")
    else:
        print("✗ Failed to patch Replit Auth")
    
    # Update route files
    if update_route_files():
        print("✓ Successfully updated route files")
    else:
        print("✗ Failed to update route files")