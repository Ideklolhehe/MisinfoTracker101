"""
Prometheus metrics integration for the CIVILIAN system.
Provides standardized metrics collection and export capabilities.
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Any, Union

from prometheus_client import Counter, Gauge, Histogram, Summary, Info, REGISTRY
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from flask import Response, Blueprint

from utils.environment import ENABLE_PROMETHEUS

# Configure module logger
logger = logging.getLogger(__name__)

# Namespace for metrics
METRICS_NAMESPACE = "civilian"

# Counters
narrative_counter = Counter(
    f"{METRICS_NAMESPACE}_narratives_total", 
    "Total narratives processed",
    ["status", "source"]
)

clustering_counter = Counter(
    f"{METRICS_NAMESPACE}_clustering_operations_total",
    "Total clustering operations performed",
    ["algorithm"]
)

detection_counter = Counter(
    f"{METRICS_NAMESPACE}_detections_total",
    "Total misinformation detections",
    ["confidence_level"]
)

verification_counter = Counter(
    f"{METRICS_NAMESPACE}_verifications_total",
    "Total content verification requests",
    ["type", "status"]
)

api_counter = Counter(
    f"{METRICS_NAMESPACE}_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"]
)

error_counter = Counter(
    f"{METRICS_NAMESPACE}_errors_total",
    "Total errors",
    ["component", "error_type"]
)

# Gauges
active_narratives_gauge = Gauge(
    f"{METRICS_NAMESPACE}_active_narratives",
    "Number of active narratives",
    ["status"]
)

cluster_count_gauge = Gauge(
    f"{METRICS_NAMESPACE}_cluster_count",
    "Number of narrative clusters",
    ["algorithm"]
)

silhouette_gauge = Gauge(
    f"{METRICS_NAMESPACE}_silhouette_score",
    "Silhouette score for clusters",
    ["algorithm"]
)

threat_level_gauge = Gauge(
    f"{METRICS_NAMESPACE}_narrative_threat_level",
    "Threat level of narratives",
    ["narrative_id"]
)

queue_size_gauge = Gauge(
    f"{METRICS_NAMESPACE}_queue_size",
    "Size of processing queues",
    ["queue_name"]
)

# Histograms
processing_time_histogram = Histogram(
    f"{METRICS_NAMESPACE}_processing_time_seconds",
    "Time spent processing operations",
    ["operation"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
)

confidence_histogram = Histogram(
    f"{METRICS_NAMESPACE}_confidence_scores",
    "Distribution of confidence scores",
    ["operation"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
)

# Summaries
api_latency_summary = Summary(
    f"{METRICS_NAMESPACE}_api_latency_seconds",
    "API endpoint latency in seconds",
    ["endpoint"]
)

# Information
system_info = Info(
    f"{METRICS_NAMESPACE}_system_info",
    "Information about the CIVILIAN system"
)

# Set system information
system_info.info({
    "version": "1.0.0",
    "description": "CIVILIAN AI-powered misinformation detection system"
})


class MetricsMiddleware:
    """
    Middleware to collect metrics on HTTP requests.
    """
    
    def __init__(self, app):
        """
        Initialize the middleware.
        
        Args:
            app: Flask application
        """
        if not ENABLE_PROMETHEUS:
            return
            
        self.app = app
        
        # Wrap the dispatch request to measure latencies
        original_dispatch = app.dispatch_request
        
        def metrics_dispatch_wrapper():
            start_time = time.time()
            try:
                return original_dispatch()
            finally:
                # Record request metrics on completion
                latency = time.time() - start_time
                path = app.request.path
                method = app.request.method
                
                # Skip metrics endpoint itself
                if path != "/metrics":
                    api_latency_summary.labels(endpoint=path).observe(latency)
                    
        app.dispatch_request = metrics_dispatch_wrapper


class PerformanceTimer:
    """
    Context manager for timing operations and recording metrics.
    """
    
    def __init__(self, operation: str):
        """
        Initialize the timer.
        
        Args:
            operation: Name of the operation being timed
        """
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not ENABLE_PROMETHEUS:
            return
            
        end_time = time.time()
        latency = end_time - self.start_time
        
        # Record metrics
        processing_time_histogram.labels(operation=self.operation).observe(latency)
        
        if exc_type is not None:
            # Record error
            error_counter.labels(
                component=self.operation,
                error_type=exc_type.__name__
            ).inc()


def create_metrics_blueprint() -> Blueprint:
    """
    Create a Flask Blueprint for serving Prometheus metrics.
    
    Returns:
        Flask Blueprint
    """
    metrics_bp = Blueprint("metrics", __name__)
    
    @metrics_bp.route("/metrics")
    def metrics():
        """Serve Prometheus metrics."""
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
        
    return metrics_bp


def increment_counter(
    counter: Counter, 
    labels: Dict[str, str] = None,
    value: float = 1.0
) -> None:
    """
    Increment a counter with optional labels.
    
    Args:
        counter: The counter to increment
        labels: Dictionary of label values
        value: Amount to increment by
    """
    if not ENABLE_PROMETHEUS:
        return
        
    if labels:
        counter.labels(**labels).inc(value)
    else:
        counter.inc(value)


def set_gauge(
    gauge: Gauge, 
    value: float,
    labels: Dict[str, str] = None
) -> None:
    """
    Set a gauge value with optional labels.
    
    Args:
        gauge: The gauge to set
        value: Value to set
        labels: Dictionary of label values
    """
    if not ENABLE_PROMETHEUS:
        return
        
    if labels:
        gauge.labels(**labels).set(value)
    else:
        gauge.set(value)


def observe_histogram(
    histogram: Histogram,
    value: float,
    labels: Dict[str, str] = None
) -> None:
    """
    Observe a value for a histogram with optional labels.
    
    Args:
        histogram: The histogram to observe
        value: Value to observe
        labels: Dictionary of label values
    """
    if not ENABLE_PROMETHEUS:
        return
        
    if labels:
        histogram.labels(**labels).observe(value)
    else:
        histogram.observe(value)


def time_operation(operation: str) -> PerformanceTimer:
    """
    Create a timer for measuring operation duration.
    
    Args:
        operation: Name of the operation
        
    Returns:
        PerformanceTimer context manager
    """
    return PerformanceTimer(operation)