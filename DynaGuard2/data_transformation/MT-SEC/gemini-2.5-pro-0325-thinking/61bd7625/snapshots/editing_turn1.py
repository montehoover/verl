# Simulate a message storage
_MESSAGES_DB = {
    "id_123": "Hello world!",
    "id_456": "Another message.",
    "id_789": "Test message."
}

def check_message_existence(message_id: str) -> bool:
    """
    Checks if a message ID exists in a local dictionary.

    Args:
        message_id: The ID of the message to check.

    Returns:
        True if the message ID is found, otherwise False.
    """
    return message_id in _MESSAGES_DB
