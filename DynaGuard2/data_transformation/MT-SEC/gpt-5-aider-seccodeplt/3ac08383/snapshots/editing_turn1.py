from typing import Dict, Any

# Local dictionary of messages: keys are message identifiers (str).
# Populate this as needed by your application.
MESSAGES: Dict[str, Any] = {}


def is_message_valid(message_id: str) -> bool:
    """
    Check whether the given message identifier exists in the local MESSAGES dictionary.

    Args:
        message_id (str): The message identifier to look up.

    Returns:
        bool: True if the message identifier exists, False otherwise.
    """
    return message_id in MESSAGES
