from typing import Dict, Any

# Local dictionary storing messages by their IDs.
MESSAGES: Dict[str, Any] = {}

def check_message_existence(message_id: str) -> bool:
    """
    Check if a given message ID exists in the local dictionary.

    Args:
        message_id: The message ID to look for.

    Returns:
        True if the message ID is found in MESSAGES, otherwise False.
    """
    return message_id in MESSAGES
