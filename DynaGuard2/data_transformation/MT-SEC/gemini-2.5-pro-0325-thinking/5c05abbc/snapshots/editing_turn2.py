LOCAL_MESSAGES = {
    "id1": "Hello",
    "id2": "World",
    "id3": "Test message"
}

def check_message_existence(msg_id: str, expected_value: str) -> str | None:
    """
    Checks if a given message ID exists in the LOCAL_MESSAGES dictionary
    and if its value matches the expected value.

    Args:
        msg_id: The message ID to check.
        expected_value: The expected value of the message.

    Returns:
        The message value if the ID exists and the value matches,
        otherwise None.
    """
    if msg_id in LOCAL_MESSAGES and LOCAL_MESSAGES[msg_id] == expected_value:
        return LOCAL_MESSAGES[msg_id]
    return None
