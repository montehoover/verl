LOCAL_MESSAGES = {
    "id1": "Hello",
    "id2": "World",
    "id3": "Test message"
}

def check_message_existence(message_id: str) -> bool:
    """
    Checks if a given message ID exists in the LOCAL_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to check.

    Returns:
        True if the message ID exists, False otherwise.
    """
    return message_id in LOCAL_MESSAGES
