# Example API_MESSAGES dictionary for demonstration
API_MESSAGES = {
    "msg_001": "Welcome to our service!",
    "msg_002": "Your request has been processed successfully.",
    "msg_003": "Error: Invalid input provided.",
    "msg_004": "Thank you for using our API.",
    "msg_005": "Please check your credentials and try again."
}

def fetch_api_message(message_id):
    """
    Retrieve message content from the API_MESSAGES dictionary.
    
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
