def check_message_existence(message_id: str) -> bool:
    """
    Check if a message ID exists in a simulated local message storage.

    Args:
        message_id (str): The ID of the message to check.

    Returns:
        bool: True if the message ID exists, False otherwise.
    """
    # Simulated local message storage using a dictionary
    message_store = {
        "msg-001": {"text": "Hello"},
        "msg-002": {"text": "Hi"},
        "msg-003": {"text": "Hey"},
    }

    return message_id in message_store
