from flask import render_template
from app import app
import logging

logger = logging.getLogger(__name__)

@app.route('/test', methods=['GET'])
def test():
    """
    A simple test route to verify that the Flask application is working correctly.
    """
    try:
        logger.info("Rendering test page")
        return render_template('test.html')
    except Exception as e:
        logger.error(f"Error in test page endpoint: {e}")
        return f"Error: {str(e)}", 500