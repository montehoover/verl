from typing import Dict
import hashlib

# Example API messages dictionary for demonstration
API_MESSAGES: Dict[str, str] = {
    "msg_001": "Welcome to the system.",
    "msg_002": "Your request has been received.",
    "msg_003": "An error occurred. Please try again.",
}


def fetch_api_message(message_id: str) -> str:
    """
    Retrieve the message content from the API_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to retrieve.

    Returns:
        The content of the message corresponding to the given message_id.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message ID not found: {message_id}") from None


def check_signature(message: str, signature: str) -> bool:
    """
    Compute a SHA-256 hash of the message and compare it to the provided signature.

    Args:
        message: The message content as a string.
        signature: The expected signature (hex string).

    Returns:
        True if the computed hash matches the provided signature, otherwise False.
        Returns False if unsupported inputs or operations are detected.
    """
    try:
        if not isinstance(message, str) or not isinstance(signature, str):
            return False
        computed_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        return computed_hash == signature.lower()
    except Exception:
        # Any unexpected error or unsupported operation results in a False return
        return False


if __name__ == "__main__":
    # Example usage:
    print(fetch_api_message("msg_001"))  # Expected: "Welcome to the system."
    try:
        print(fetch_api_message("msg_999"))  # This will raise a KeyError
    except KeyError as e:
        print(e)

    # Signature checking examples:
    message = "Welcome to the system."
    correct_signature = hashlib.sha256(message.encode("utf-8")).hexdigest()
    print(check_signature(message, correct_signature))  # Expected: True
    print(check_signature(message, "invalid_signature"))  # Expected: False
