from typing import Dict
import hashlib

# Example API messages dictionary for demonstration
API_MESSAGES: Dict[str, str] = {
    "msg_001": "Welcome to the system.",
    "msg_002": "Your request has been received.",
    "msg_003": "An error occurred. Please try again.",
}

# Expected cryptographic signatures for specific API messages
VALID_API_MESSAGES: Dict[str, str] = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
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


def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Verify the legitimacy of a message by checking that the provided cryptographic
    signature matches the expected signature for the given API message ID.

    Args:
        api_message_id: The ID of the API message.
        crypto_signature: The cryptographic signature to verify.

    Returns:
        True if the signature matches the expected value for the given ID, otherwise False.
        Returns False for unsupported inputs or on any error.
    """
    try:
        if not isinstance(api_message_id, str) or not isinstance(crypto_signature, str):
            return False
        expected_signature = VALID_API_MESSAGES.get(api_message_id)
        if expected_signature is None:
            return False
        return crypto_signature.lower() == expected_signature.lower()
    except Exception:
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

    # verify_message examples (based on VALID_API_MESSAGES mapping):
    print(verify_message("msg_1001", "abcd1234efgh5678ijkl9012mnop3456"))  # Expected: True
    print(verify_message("msg_1002", "wrong_signature"))  # Expected: False
    print(verify_message("unknown_id", "anything"))  # Expected: False
