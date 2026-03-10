def check_message_existence(msg_id: str) -> bool:
    """
    Check if the given message ID exists in the LOCAL_MESSAGES dictionary.

    Args:
        msg_id: The message ID to check.

    Returns:
        True if the message ID is found in LOCAL_MESSAGES, otherwise False.
    """
    local_messages = globals().get('LOCAL_MESSAGES', {})
    return msg_id in local_messages
