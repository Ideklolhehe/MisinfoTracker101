"""
Simplified handler for Vercel serverless functions
"""

# Simple HTML with Speed Insights
html = """
<!DOCTYPE html>
<html>
<head>
    <title>CIVILIAN - Speed Insights</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script defer src="https://cdn.vercel-insights.com/v1/speed-insights/script.js"></script>
</head>
<body style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:20px;text-align:center">
    <h1>CIVILIAN</h1>
    <p>Speed Insights is now active</p>
    <p style="margin-top:40px;color:#666">The full system is running on Replit</p>
</body>
</html>
"""

def handler(request):
    """
    This function is the entry point for the Vercel serverless function.
    It returns a simple HTML page with Speed Insights enabled.
    """
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html",
        },
        "body": html
    }