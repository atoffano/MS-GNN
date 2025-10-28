"""Utility decorator for timing function execution.

This module provides a simple timing decorator to measure and log the execution
time of functions, useful for performance profiling during data processing and
model training.
"""

import time

def timeit(func):
    """Decorator to measure and print function execution time.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function that prints execution time when called
    """
    def wrapper(*args, **kwargs):
        """Wrapper function that times the decorated function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper
