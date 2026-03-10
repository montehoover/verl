from typing import Optional

def check_message_existence(msg_id: str, expected_value: str) -> Optional[str]:
    """
    Verify that the given message ID exists in the LOCAL_MESSAGES dictionary and that
    the associated value matches the expected value.

    Args:
        msg_id: The message ID to check.
        expected_value: The value to compare against the stored message value.

    Returns:
        The message value if the ID exists and the value matches, otherwise None.
    """
    local_messages = globals().get('LOCAL_MESSAGES', {})
    value = local_messages.get(msg_id)
    if value == expected_value:
        return value
    return None
