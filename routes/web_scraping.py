"""
Routes for web scraping functionality in the CIVILIAN system.
"""

import logging
import json
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required
from urllib.parse import urlparse
import threading

from services.web_scraping_service import web_scraping_service
from data_sources.web_source_manager import web_source_manager
from utils.web_scraper import WebScraper

# Configure logger
logger = logging.getLogger(__name__)

# Create blueprint
web_scraping_bp = Blueprint("web_scraping", __name__, url_prefix="/web-scraping")

# Initialize web scraper for route handlers
web_scraper = WebScraper()


@web_scraping_bp.route("/")
@login_required
def index():
    """Web scraping dashboard page."""
    return render_template("web_scraping/dashboard.html")


@web_scraping_bp.route("/scan", methods=["GET", "POST"])
@login_required
def scan_url():
    """Scan a URL and analyze its content."""
    if request.method == "POST":
        data = request.get_json() or {}
        url = data.get("url", "")
        depth = int(data.get("depth", 1))
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
            
        # Validate URL format
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
            
        try:
            # Run scan in background thread
            scan_id = web_source_manager.add_url_job(url, "single" if depth <= 1 else "crawl", 
                                                 {"max_pages": depth, "source_name": f"Manual Scan: {url}"})
            
            # Return immediate response with job ID
            return jsonify({
                "scan_id": scan_id,
                "url": url,
                "depth": depth,
                "status": "processing"
            })
        except Exception as e:
            logger.error(f"Error scanning URL: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # GET request - render scan form
        return render_template("web_scraping/scan.html")


@web_scraping_bp.route("/scan/<scan_id>")
@login_required
def scan_status(scan_id):
    """Get the status of a scan job."""
    try:
        status = web_source_manager.get_job_status(scan_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting scan status: {e}")
        return jsonify({"error": str(e)}), 500


@web_scraping_bp.route("/search", methods=["GET", "POST"])
@login_required
def search():
    """Search for content and monitor results."""
    if request.method == "POST":
        data = request.get_json() or {}
        search_term = data.get("search_term", "")
        limit = int(data.get("limit", 10))
        
        if not search_term:
            return jsonify({"error": "Search term is required"}), 400
            
        try:
            # Run search and monitoring in background thread
            def run_search():
                web_scraping_service.search_and_monitor(search_term, limit)
                
            thread = threading.Thread(target=run_search)
            thread.daemon = True
            thread.start()
            
            # Return immediate response
            return jsonify({
                "search_term": search_term,
                "limit": limit,
                "status": "processing"
            })
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # GET request - render search form
        return render_template("web_scraping/search.html")


@web_scraping_bp.route("/monitoring", methods=["GET", "POST"])
@login_required
def monitoring():
    """Manage web monitoring configuration."""
    if request.method == "POST":
        data = request.get_json() or {}
        action = data.get("action", "")
        
        if action == "add_domain":
            domain = data.get("domain", "")
            category = data.get("category", "other")
            priority = int(data.get("priority", 2))
            
            if not domain:
                return jsonify({"error": "Domain is required"}), 400
                
            success = web_scraping_service.add_focused_domain(domain, category, priority)
            return jsonify({"success": success})
            
        elif action == "add_search_term":
            term = data.get("term", "")
            category = data.get("category", "other")
            
            if not term:
                return jsonify({"error": "Search term is required"}), 400
                
            success = web_scraping_service.add_search_term(term, category)
            return jsonify({"success": success})
            
        elif action == "start_monitoring":
            web_scraping_service.start_scheduled_scraping()
            return jsonify({"success": True, "status": "started"})
            
        elif action == "stop_monitoring":
            web_scraping_service.stop_scheduled_scraping()
            return jsonify({"success": True, "status": "stopped"})
            
        else:
            return jsonify({"error": "Unknown action"}), 400
    else:
        # GET request - render monitoring configuration
        domain_stats = web_scraping_service.get_domain_stats()
        search_term_stats = web_scraping_service.get_search_term_stats()
        
        return render_template("web_scraping/monitoring.html", 
                              domain_stats=domain_stats,
                              search_term_stats=search_term_stats)


@web_scraping_bp.route("/register-source", methods=["GET", "POST"])
@login_required
def register_source():
    """Register a new web source for monitoring."""
    if request.method == "POST":
        data = request.get_json() or {}
        
        # Validate required fields
        for field in ["name", "url", "source_type"]:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Format source data for registration
        source_data = {
            "name": data["name"],
            "url": data["url"],
            "source_type": data["source_type"],
            "is_active": data.get("is_active", True)
        }
        
        # Add config if provided
        if "config" in data:
            source_data["config"] = data["config"]
        else:
            # Create default config based on source type
            config = {"url": data["url"]}
            
            if data["source_type"] == "search":
                config["search_term"] = data.get("search_term", "")
                config["search_engine"] = data.get("search_engine", "bing")
                config["limit"] = data.get("limit", 10)
                
            source_data["config"] = config
        
        try:
            # Register the source
            source_id = web_source_manager.register_source(source_data)
            
            if source_id:
                return jsonify({"success": True, "source_id": source_id})
            else:
                return jsonify({"error": "Failed to register source"}), 500
        except Exception as e:
            logger.error(f"Error registering source: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # GET request - render source registration form
        return render_template("web_scraping/register_source.html")


@web_scraping_bp.route("/sources")
@login_required
def list_sources():
    """List all web sources."""
    try:
        sources = []
        db_sources = web_source_manager.get_active_sources()
        
        for source in db_sources:
            sources.append({
                "id": source.id,
                "name": source.name,
                "source_type": source.source_type,
                "is_active": source.is_active,
                "last_ingestion": source.last_ingestion.isoformat() if source.last_ingestion else None
            })
            
        return jsonify({"sources": sources})
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        return jsonify({"error": str(e)}), 500


@web_scraping_bp.route("/sources/<int:source_id>", methods=["GET", "PUT"])
@login_required
def manage_source(source_id):
    """Manage a specific web source."""
    source = web_source_manager.get_source_by_id(source_id)
    
    if not source:
        return jsonify({"error": "Source not found"}), 404
    
    if request.method == "PUT":
        data = request.get_json() or {}
        
        # Update active status if provided
        if "is_active" in data:
            success = web_source_manager.update_source_status(source_id, data["is_active"])
            if not success:
                return jsonify({"error": "Failed to update source status"}), 500
        
        return jsonify({"success": True})
    else:
        # GET request - return source details
        try:
            config = json.loads(source.config) if source.config else {}
            meta_data = json.loads(source.meta_data) if source.meta_data else {}
            
            return jsonify({
                "id": source.id,
                "name": source.name,
                "source_type": source.source_type,
                "is_active": source.is_active,
                "last_ingestion": source.last_ingestion.isoformat() if source.last_ingestion else None,
                "config": config,
                "meta_data": meta_data
            })
        except Exception as e:
            logger.error(f"Error getting source details: {e}")
            return jsonify({"error": str(e)}), 500


@web_scraping_bp.route("/status")
@login_required
def service_status():
    """Get the status of the web scraping service."""
    try:
        domain_stats = web_scraping_service.get_domain_stats()
        search_term_stats = web_scraping_service.get_search_term_stats()
        
        return jsonify({
            "domain_stats": domain_stats,
            "search_term_stats": search_term_stats,
            "is_monitoring_active": web_scraping_service.is_running
        })
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        return jsonify({"error": str(e)}), 500