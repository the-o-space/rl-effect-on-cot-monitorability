"""
Centralized Logfire logging configuration.

Import and use `logger` throughout the project for consistent logging.
"""
import os
import logfire

# Configure Logfire once on import
# Note: LOGFIRE_TOKEN and LOGFIRE_ENVIRONMENT are the standard env vars
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    environment=os.getenv("LOGFIRE_ENVIRONMENT"),
)

# Export the configured logfire instance
logger = logfire

__all__ = ["logger"]
