API_MESSAGES = {
    "msg_1": "Welcome to the API!",
    "msg_2": "Your request is being processed.",
    "msg_3": "Operation completed successfully.",
}


def fetch_message(message_id: str) -> str:
    """
    Retrieve a message's content by ID from the local API_MESSAGES dictionary.

    Args:
        message_id: The identifier of the message to fetch.

    Returns:
        The message content as a string.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError as e:
        raise KeyError(f"Message ID '{message_id}' not found.") from e
