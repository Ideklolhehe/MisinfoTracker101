from datetime import datetime
from app import app

@app.template_filter('strftime')
def _jinja2_filter_datetime(timestamp):
    """Convert a Unix timestamp to a formatted date string."""
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@app.template_filter('tojson')
def _jinja2_filter_tojson(obj):
    """Convert an object to a JSON string."""
    import json
    return json.dumps(obj)