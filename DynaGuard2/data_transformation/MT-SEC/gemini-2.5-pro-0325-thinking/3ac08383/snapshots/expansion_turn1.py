API_MESSAGES = {
    "id1": "Hello, world!",
    "id2": "This is a test message.",
    "id3": "Another message here.",
}

def fetch_api_message(message_id: str) -> str:
    """
    Retrieves a message from the API_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to retrieve.

    Returns:
        The content of the message.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message with ID '{message_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        print(fetch_api_message("id1"))
        print(fetch_api_message("id2"))
        # This will raise a KeyError
        print(fetch_api_message("id4"))
    except KeyError as e:
        print(e)
