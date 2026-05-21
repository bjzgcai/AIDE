"""
Network configuration for handling timeouts and retries.
"""
import os
import time
from typing import Optional

# Default timeout settings
DEFAULT_CONNECT_TIMEOUT = 30  # seconds
DEFAULT_READ_TIMEOUT = 60     # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0     # seconds

# Environment variable overrides
def get_connect_timeout() -> int:
    """Get connection timeout from environment or use default"""
    return int(os.getenv("HF_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT))

def get_read_timeout() -> int:
    """Get read timeout from environment or use default"""
    return int(os.getenv("HF_READ_TIMEOUT", DEFAULT_READ_TIMEOUT))

def get_max_retries() -> int:
    """Get max retries from environment or use default"""
    return int(os.getenv("HF_MAX_RETRIES", DEFAULT_MAX_RETRIES))

def get_retry_delay() -> float:
    """Get retry delay from environment or use default"""
    return float(os.getenv("HF_RETRY_DELAY", DEFAULT_RETRY_DELAY))

def exponential_backoff_delay(base_delay: float, attempt: int, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay"""
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)

def safe_request_with_retry(func, *args, max_retries: Optional[int] = None, 
                          base_delay: Optional[float] = None, **kwargs):
    """
    Execute a function with retry mechanism and exponential backoff.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        max_retries: Maximum number of retries (default from config)
        base_delay: Base delay for exponential backoff (default from config)
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or None if all retries failed
    """
    if max_retries is None:
        max_retries = get_max_retries()
    if base_delay is None:
        base_delay = get_retry_delay()
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            delay = exponential_backoff_delay(base_delay, attempt)
            time.sleep(delay)
    
    return None 