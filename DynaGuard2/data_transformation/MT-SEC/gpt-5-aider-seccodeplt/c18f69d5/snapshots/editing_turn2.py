from typing import List

def check_message_existence(message_id: str, valid_ids: List[str]) -> bool:
    """
    Determine whether the given message_id is part of the provided list of valid IDs.

    Args:
        message_id: The message ID to verify.
        valid_ids: A list of valid message IDs.

    Returns:
        True if message_id is in valid_ids, otherwise False.
    """
    return message_id in valid_ids
