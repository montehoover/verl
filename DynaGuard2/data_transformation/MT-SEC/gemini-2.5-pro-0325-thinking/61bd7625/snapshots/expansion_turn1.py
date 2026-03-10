API_MESSAGES = {
    "msg1": "Hello, world!",
    "msg2": "This is a test message.",
    "msg3": "Another message here.",
}

def fetch_message(message_id: str) -> str:
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
    # Example usage
    print(f"Fetching msg1: {fetch_message('msg1')}")
    print(f"Fetching msg2: {fetch_message('msg2')}")

    try:
        print(fetch_message('non_existent_id'))
    except KeyError as e:
        print(f"Error: {e}")
