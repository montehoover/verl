API_MESSAGES = {
    "welcome": "Welcome to the service!",
    "farewell": "Goodbye and take care!",
    "status_ok": "Your request was successful.",
    "info_42": "The answer is 42."
}

def fetch_api_message(message_id):
    """
    Retrieve a message from the API_MESSAGES dictionary by its ID.

    Args:
        message_id: The identifier for the message to fetch.

    Returns:
        The message content associated with the given message_id.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError as e:
        raise KeyError(f"Message ID '{message_id}' not found in API_MESSAGES.") from e


if __name__ == "__main__":
    # Demonstration of successful retrieval
    print("Existing ID:", fetch_api_message("welcome"))

    # Demonstration of missing ID causing a KeyError
    try:
        print("Missing ID:", fetch_api_message("does_not_exist"))
    except KeyError as err:
        print("Error:", err)
