# routes/__init__.py
# Import blueprints for registration
from .home import home_bp
from .profile import profile_bp

# Add other blueprints as they are created
blueprints = [
    home_bp,
    profile_bp,
]

# This allows blueprints to be imported in the application