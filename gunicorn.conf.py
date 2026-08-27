"""Gunicorn settings for Render's CPU-only TensorFlow service."""

import os


bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
threads = 1
timeout = 120
graceful_timeout = 30
keepalive = 5
