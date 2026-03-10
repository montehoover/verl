# Example API messages dictionary
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "An error occurred. Please try again later.",
    "msg_004": "Thank you for using our API.",
    "msg_005": "Invalid input provided.",
    "msg_006": "Authentication required.",
    "msg_007": "Rate limit exceeded.",
    "msg_008": "Resource not found.",
    "msg_009": "Server maintenance in progress.",
    "msg_010": "Operation completed successfully."
}


def fetch_api_message(message_id):
    """
    Fetch a message from the API_MESSAGES dictionary by its ID.
    
    Args:
        message_id (str): The ID of the message to retrieve.
        
    Returns:
        str: The message content.
        
    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    if message_id not in API_MESSAGES:
        raise KeyError(f"Message ID '{message_id}' not found in API_MESSAGES")
    
    return API_MESSAGES[message_id]


# Example usage
if __name__ == "__main__":
    # Test successful retrieval
    try:
        message = fetch_api_message("msg_001")
        print(f"Retrieved message: {message}")
    except KeyError as e:
        print(f"Error: {e}")
    
    # Test with non-existent ID
    try:
        message = fetch_api_message("msg_999")
        print(f"Retrieved message: {message}")
    except KeyError as e:
        print(f"Error: {e}")
