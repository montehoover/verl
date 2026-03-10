import hashlib

API_MESSAGES = {
    "msg_1": "Welcome to the API!",
    "msg_2": "Your request is being processed.",
    "msg_3": "Operation completed successfully.",
}

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
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


def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Authenticate a message by verifying its signature against the computed SHA-256 digest
    and, when available, against an expected signature registry.

    Args:
        message_id: The identifier of the message to authenticate.
        signature: The provided signature to validate.

    Returns:
        True if the message's signature is valid; False otherwise.
    """
    # Basic type validation
    if not isinstance(message_id, str) or not isinstance(signature, str):
        return False

    # Attempt to fetch the message content
    try:
        content = fetch_message(message_id)
    except KeyError:
        return False

    # Verify signature matches computed digest of the content
    if not verify_message_signature(content, signature):
        return False

    # If we have a registry entry for this message_id, enforce it matches too
    expected_sig = VALID_API_MESSAGES.get(message_id)
    if expected_sig is not None and signature != expected_sig:
        return False

    return True
