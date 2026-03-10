from typing import Optional

# Simulate a message storage with signatures
_MESSAGES_DB = {
    "id_123": {"message": "Hello world!", "signature": "sig_abc"},
    "id_456": {"message": "Another message.", "signature": "sig_def"},
    "id_789": {"message": "Test message.", "signature": "sig_ghi"}
}

def check_message_existence(message_id: str, signature: str) -> Optional[str]:
    """
    Checks if a message ID exists and the provided signature matches.

    Args:
        message_id: The ID of the message to check.
        signature: The signature to validate against the stored signature.

    Returns:
        The message_id if both the message ID exists and the signature matches,
        otherwise None.
    """
    if message_id in _MESSAGES_DB:
        stored_data = _MESSAGES_DB[message_id]
        if stored_data.get("signature") == signature:
            return message_id
    return None
