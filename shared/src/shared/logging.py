"""
Centralized Logfire logging configuration.

Import and use `logger` throughout the project for consistent logging.
"""
import os
import logfire

# Configure Logfire once on import
# Note: LOGFIRE_TOKEN env var is the standard way to authenticate
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="rl-cot-monitorability",
)

# Export the configured logfire instance
logger = logfire

__all__ = ["logger"]
