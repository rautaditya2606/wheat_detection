import os

# Gunicorn configuration file
bind = "0.0.0.0:5000"
workers = 2
threads = 4
timeout = 120
