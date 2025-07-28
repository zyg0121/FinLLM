import time
import functools
import threading
import logging
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from flask import request, Flask, Blueprint

# Configure logging
logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter('finllm_request_count', 'Total number of requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('finllm_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('finllm_prediction_count', 'Number of predictions made', ['ticker', 'cache_hit'])
PREDICTION_ERROR_COUNT = Counter('finllm_prediction_error_count', 'Number of prediction errors', ['ticker'])
PREDICTION_LATENCY = Histogram('finllm_prediction_latency_seconds', 'Prediction latency in seconds', ['ticker'])
MODEL_LOAD_TIME = Gauge('finllm_model_load_time_seconds', 'Time taken to load the model')
ACTIVE_REQUESTS = Gauge('finllm_active_requests', 'Number of active requests')
MEMORY_USAGE = Gauge('finllm_memory_usage_bytes', 'Memory usage in bytes')
GPU_MEMORY_USAGE = Gauge('finllm_gpu_memory_usage_bytes', 'GPU memory usage in bytes')
CACHE_SIZE = Gauge('finllm_cache_size', 'Number of items in prediction cache')
CACHE_HIT_RATIO = Gauge('finllm_cache_hit_ratio', 'Cache hit ratio')
BATCH_SIZE_HISTOGRAM = Histogram('finllm_batch_size', 'Batch size for batch predictions')

# Create a blueprint for metrics endpoints
metrics_blueprint = Blueprint('metrics', __name__)

def start_metrics_server(port=8000):
    """Start a separate metrics server on the specified port"""
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")

def track_request_count(func):
    """Decorator to track request count and latency"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ACTIVE_REQUESTS.inc()
        
        try:
            response = func(*args, **kwargs)
            status = response[1] if isinstance(response, tuple) else 200
        except Exception as e:
            status = 500
            raise e
        finally:
            ACTIVE_REQUESTS.dec()
            latency = time.time() - start_time
            endpoint = request.path
            method = request.method
            
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
        
        return response
    
    return wrapper

def track_memory_usage():
    """Function to periodically update memory usage metrics"""
    try:
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        MEMORY_USAGE.set(memory_info.rss)
    except ImportError:
        logger.warning("psutil not installed, can't track memory usage")

def track_gpu_memory():
    """Function to track GPU memory usage if available"""
    try:
        import torch
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.set(allocated)
    except ImportError:
        logger.warning("PyTorch not installed, can't track GPU memory")

def update_metrics_periodically(interval=15):
    """Periodically update various metrics"""
    while True:
        track_memory_usage()
        track_gpu_memory()
        time.sleep(interval)

def setup_monitoring(app):
    """Set up monitoring for a Flask app"""
    # Add decorator to all endpoints
    for endpoint, view_func in app.view_functions.items():
        app.view_functions[endpoint] = track_request_count(view_func)
    
    # Start metrics updater thread
    threading.Thread(
        target=update_metrics_periodically, 
        daemon=True
    ).start()
    
    return app

def record_prediction(ticker, latency, cache_hit=False, error=False):
    """Record metrics for a prediction"""
    PREDICTION_COUNT.labels(ticker=ticker, cache_hit=str(cache_hit)).inc()
    
    if error:
        PREDICTION_ERROR_COUNT.labels(ticker=ticker).inc()
    else:
        PREDICTION_LATENCY.labels(ticker=ticker).observe(latency)

def record_model_load_time(load_time):
    """Record model load time"""
    MODEL_LOAD_TIME.set(load_time)

def update_cache_metrics(cache_size, hit_count, miss_count):
    """Update cache metrics"""
    CACHE_SIZE.set(cache_size)
    
    total = hit_count + miss_count
    if total > 0:
        CACHE_HIT_RATIO.set(hit_count / total)

@metrics_blueprint.route('/metrics')
def metrics():
    """Endpoint to expose metrics (in addition to the separate metrics server)"""
    from prometheus_client import generate_latest
    return generate_latest()