from typing import List

# Global list to store logged messages
MESSAGE_LOG: List[str] = []


def log_message(message: str) -> bool:
    """
    Append a message to the global message log.

    Args:
        message (str): The message to log.

    Returns:
        bool: True if the message was logged successfully, False otherwise.
    """
    if not isinstance(message, str):
        return False

    try:
        MESSAGE_LOG.append(message)
        return True
    except Exception:
        return False
