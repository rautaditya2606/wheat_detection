import os

# Gunicorn configuration file
# Optimized for parallel uploads + sequential background worker
bind = "0.0.0.0:10000"
workers = 1      # Keep workers low for shared memory/model cache
worker_class = "gthread" # Use gthread for better compatibility with modern libraries (httpx/trio)
threads = 12     # Increase threads to handle I/O (parallel uploads, SSE, Uptime monitoring)
timeout = 600    # High timeout for CLIP microservice + bulk processing
keepalive = 5    # Keep connections alive for SSE stability
max_requests = 1000 # Restart worker periodically to clear memory leaks
max_requests_jitter = 50
