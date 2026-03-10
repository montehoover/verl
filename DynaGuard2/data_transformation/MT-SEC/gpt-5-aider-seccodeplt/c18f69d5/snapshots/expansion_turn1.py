from typing import Dict

# Example dictionary to simulate an external API response store
API_MESSAGES: Dict[str, str] = {
    "msg_001": "Welcome to the system!",
    "msg_002": "Your request has been processed.",
    "msg_003": "Please verify your email address.",
    "msg_100": "System maintenance scheduled at 02:00 UTC."
}


def fetch_api_message(message_id: str) -> str:
    """
    Retrieve a message content by its ID from the API_MESSAGES dictionary.

    Args:
        message_id: The unique identifier for the message to fetch.

    Returns:
        The message content as a string.

    Raises:
        KeyError: If the provided message_id does not exist in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message ID '{message_id}' not found in API_MESSAGES.") from None


if __name__ == "__main__":
    # Example usage:
    print(fetch_api_message("msg_001"))  # Expected: "Welcome to the system!"

    # This will raise a KeyError to simulate a missing message ID
    # Uncomment to test:
    # print(fetch_api_message("msg_999"))
