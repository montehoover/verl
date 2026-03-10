from typing import Dict

# Example API messages dictionary for demonstration purposes.
API_MESSAGES: Dict[str, str] = {
    "msg_1": "Hello, world!",
    "msg_2": "Your order has been shipped.",
    "msg_3": "Please verify your email address.",
    "msg_4": "Your password has been reset.",
}


def fetch_api_message(message_id: str) -> str:
    """
    Retrieve the message content for a given message_id from API_MESSAGES.

    Args:
        message_id: The identifier of the message to fetch.

    Returns:
        The message content as a string.

    Raises:
        KeyError: If the message_id does not exist in API_MESSAGES.
    """
    if message_id in API_MESSAGES:
        return API_MESSAGES[message_id]
    raise KeyError(f"Message ID not found: {message_id}")


if __name__ == "__main__":
    # Example usage:
    print(fetch_api_message("msg_1"))  # Expected: "Hello, world!"

    # Simulate a missing message ID to demonstrate KeyError
    try:
        print(fetch_api_message("unknown_id"))
    except KeyError as err:
        print(err)
