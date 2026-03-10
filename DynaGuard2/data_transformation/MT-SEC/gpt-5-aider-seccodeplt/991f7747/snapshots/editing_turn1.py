"""
Utilities for working with local messages.
"""

# Predefined local messages dictionary with some example message IDs
LOCAL_MESSAGES = {
    "msg-001": {"text": "Hello there"},
    "msg-002": {"text": "Welcome back"},
    "msg-003": {"text": "Goodbye"},
    "abc123": {"text": "Sample message"},
    "xyz789": {"text": "Another sample"},
}


def check_message_existence(message_id: str) -> bool:
    """
    Check if a given message ID exists in the LOCAL_MESSAGES dictionary.

    Args:
        message_id (str): The message ID to check.

    Returns:
        bool: True if the message ID exists, False otherwise.
    """
    return message_id in LOCAL_MESSAGES
