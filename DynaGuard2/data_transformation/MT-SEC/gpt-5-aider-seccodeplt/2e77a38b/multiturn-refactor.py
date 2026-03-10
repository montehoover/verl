"""
In-memory caching utilities for customer data.

This module exposes a process-local cache, `user_cache`, that stores
user/customer data to avoid repeated database queries during a single
process lifetime. Use `store_customer_info()` to add or update entries.

Logging:
    Logging is initialized within `store_customer_info` on first call. It logs:
    - INFO messages when caching succeeds.
    - WARNING messages for validation failures.
    - ERROR messages with stack traces for unexpected exceptions.
"""

import logging


# Global dictionary used as an in-memory cache keyed by customer ID.
user_cache = {}


def store_customer_info(cust_id: str, cust_data: dict) -> bool:
    """
    Cache customer data in the global `user_cache`.

    This function stores user-specific information in a dictionary for quick
    retrieval, reducing the need for repeated database queries.

    Args:
        cust_id (str): A unique string that identifies each user.
        cust_data (dict): A dictionary encompassing various user-related
            information.

    Returns:
        bool: True for successful caching, False for failure.

    Notes:
        - A shallow copy of `cust_data` is stored to prevent external mutations
          from affecting the cached entry.
        - The cache is process-local and will be cleared when the process
          restarts.
        - Logging is initialized on first invocation and records success/failure
          along with the customer ID.
    """
    # Initialize logging within the function (first-call setup).
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    try:
        # Validate inputs: `cust_id` must be a non-empty string and
        # `cust_data` must be a dictionary.
        if not isinstance(cust_id, str) or not cust_id:
            logger.warning(
                "Caching aborted: invalid cust_id. status=False cust_id=%r",
                cust_id,
            )
            return False

        if not isinstance(cust_data, dict):
            logger.warning(
                "Caching aborted: cust_data is not a dict. status=False cust_id=%r",
                cust_id,
            )
            return False

        # Store a shallow copy to prevent callers from mutating cached data
        # indirectly after insertion.
        user_cache[cust_id] = dict(cust_data)

        logger.info("Caching successful. status=True cust_id=%s", cust_id)
        return True

    except Exception:
        # Defensive fallback: if anything unexpected happens, log and signal failure.
        logger.exception(
            "Caching failed due to unexpected error. status=False cust_id=%r",
            cust_id,
        )
        return False
