def check_message_existence(message_id: str, signature: str) -> str | None:
    """
    Check if a message ID exists in a simulated local message storage and if the provided
    signature matches the stored signature for that message.

    Args:
        message_id (str): The ID of the message to check.
        signature (str): The signature to validate against the stored value.

    Returns:
        str | None: The message ID if it exists and the signature matches, otherwise None.
    """
    # Simulated local message storage using a dictionary
    message_store = {
        "msg-001": {"text": "Hello", "signature": "sig-001"},
        "msg-002": {"text": "Hi", "signature": "sig-002"},
        "msg-003": {"text": "Hey", "signature": "sig-003"},
    }

    record = message_store.get(message_id)
    if record and record.get("signature") == signature:
        return message_id
    return None
