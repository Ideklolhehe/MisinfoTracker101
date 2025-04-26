import re

# Path to the file
file_path = 'routes/adversarial.py'

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Remove all @login_required decorators
content = re.sub(r'@login_required\n', '', content)

# Replace current_user.id with None
content = re.sub(r'user_id=current_user\.id', 'user_id=None', content)

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.write(content)

print(f"Successfully removed @login_required decorators and fixed current_user references in {file_path}")