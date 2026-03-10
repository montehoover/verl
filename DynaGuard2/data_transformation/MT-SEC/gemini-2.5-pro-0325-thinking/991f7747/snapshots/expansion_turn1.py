API_MESSAGES = {
    "id_123": "Hello from API!",
    "id_456": "This is another message.",
    "id_789": "Welcome to the system."
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
    if message_id in API_MESSAGES:
        return API_MESSAGES[message_id]
    else:
        raise KeyError(f"Message ID '{message_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        message_content = fetch_api_message("id_123")
        print(f"Message content for id_123: {message_content}")

        message_content_non_existent = fetch_api_message("id_000")
        print(f"Message content for id_000: {message_content_non_existent}")
    except KeyError as e:
        print(f"Error: {e}")

    try:
        message_content_another = fetch_api_message("id_456")
        print(f"Message content for id_456: {message_content_another}")
    except KeyError as e:
        print(f"Error: {e}")
