LOCAL_MESSAGES = {
    "id1": "Hello",
    "id2": "World",
    "id3": "Test message"
}

def check_message_existence(message_id: str, expected_value: str) -> bool:
    """
    Checks if a given message ID exists in the LOCAL_MESSAGES dictionary
    and if its value matches the expected value.

    Args:
        message_id: The ID of the message to check.
        expected_value: The expected value of the message.

    Returns:
        True if the message ID exists and its value matches the expected value,
        False otherwise.
    """
    return message_id in LOCAL_MESSAGES and LOCAL_MESSAGES[message_id] == expected_value
