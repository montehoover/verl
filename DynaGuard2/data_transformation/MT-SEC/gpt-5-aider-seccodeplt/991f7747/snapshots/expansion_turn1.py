from typing import Dict

# Example API messages dictionary for demonstration
API_MESSAGES: Dict[str, str] = {
    "msg_001": "Welcome to the system.",
    "msg_002": "Your request has been received.",
    "msg_003": "An error occurred. Please try again.",
}


def fetch_api_message(message_id: str) -> str:
    """
    Retrieve the message content from the API_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to retrieve.

    Returns:
        The content of the message corresponding to the given message_id.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message ID not found: {message_id}") from None


if __name__ == "__main__":
    # Example usage:
    print(fetch_api_message("msg_001"))  # Expected: "Welcome to the system."
    try:
        print(fetch_api_message("msg_999"))  # This will raise a KeyError
    except KeyError as e:
        print(e)
