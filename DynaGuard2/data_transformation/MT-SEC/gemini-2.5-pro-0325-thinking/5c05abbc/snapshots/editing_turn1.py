LOCAL_MESSAGES = {
    "id1": "Hello",
    "id2": "World",
    "id3": "Test message"
}

def check_message_existence(msg_id: str) -> bool:
    """
    Checks if a given message ID exists in the LOCAL_MESSAGES dictionary.

    Args:
        msg_id: The message ID to check.

    Returns:
        True if the message ID is found, False otherwise.
    """
    return msg_id in LOCAL_MESSAGES
