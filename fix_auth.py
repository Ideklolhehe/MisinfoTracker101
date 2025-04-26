#!/usr/bin/env python
"""
Script to temporarily fix authentication issues in the CIVILIAN system.
This should be used during development to disable login requirements.
"""

import re
import os
import glob

def fix_adversarial_route():
    """Remove login_required decorators from adversarial route."""
    # Path to the file
    file_path = 'routes/adversarial.py'
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return
    
    print(f"Processing {file_path}...")
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Remove all @login_required decorators
    old_content = content
    content = re.sub(r'@login_required\n', '', content)
    
    # Replace current_user.id with None
    content = re.sub(r'user_id=current_user\.id', 'user_id=None', content)
    
    # Write the modified content back to the file if changed
    if content != old_content:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"- Fixed login_required decorators and current_user references")
    else:
        print(f"- No changes needed")

def fix_route_imports():
    """Add missing flask_login imports to route files."""
    # Find all route files
    route_files = glob.glob('routes/*.py')
    
    for file_path in route_files:
        if not os.path.exists(file_path):
            continue
            
        print(f"Processing {file_path}...")
        
        # Read the file content
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Check if login_required is used but not imported
        if 'login_required' in content and 'from flask_login import' not in content:
            if 'from flask import' in content:
                # Add import to existing flask import
                content = re.sub(
                    r'from flask import (.*)',
                    r'from flask import \1\nfrom flask_login import login_required, current_user',
                    content
                )
            else:
                # Add new import at the top of the file
                content = re.sub(
                    r'(import .*?\n\n|import .*?\n|#!/usr/bin/env python\n)',
                    r'\1from flask_login import login_required, current_user\n\n',
                    content,
                    count=1
                )
                
            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.write(content)
            print(f"- Added flask_login imports")
        else:
            print(f"- No changes needed")

def main():
    """Main function to fix authentication issues."""
    print("Fixing authentication issues in CIVILIAN system...")
    
    # Fix adversarial route login requirements
    fix_adversarial_route()
    
    # Fix route imports
    fix_route_imports()
    
    print("Authentication fixes complete!")

if __name__ == "__main__":
    main()