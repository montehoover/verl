import hashlib

API_MESSAGES = {
    "msg_1": "Welcome to the API!",
    "msg_2": "Your request is being processed.",
    "msg_3": "Operation completed successfully.",
}


def fetch_message(message_id: str) -> str:
    """
    Retrieve a message's content by ID from the local API_MESSAGES dictionary.

    Args:
        message_id: The identifier of the message to fetch.

    Returns:
        The message content as a string.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError as e:
        raise KeyError(f"Message ID '{message_id}' not found.") from e


def verify_message_signature(message_content: str, signature: str) -> bool:
    """
    Verify the authenticity of a message by comparing its SHA-256 hash to a provided signature.

    Args:
        message_content: The message content as a string.
        signature: The expected hex-encoded SHA-256 digest string.

    Returns:
        True if the computed SHA-256 hash of message_content matches the signature.
        False if the signature does not match, inputs are invalid, or an error occurs.
    """
    # Validate inputs
    if not isinstance(message_content, str) or not isinstance(signature, str):
        return False

    # Validate signature as a hex-encoded SHA-256 digest (64 hex characters)
    if len(signature) != 64:
        return False
    try:
        bytes.fromhex(signature)
    except ValueError:
        return False

    # Compute digest and compare
    try:
        computed_digest = hashlib.sha256(message_content.encode("utf-8")).hexdigest()
    except Exception:
        return False

    return computed_digest == signature.lower()
