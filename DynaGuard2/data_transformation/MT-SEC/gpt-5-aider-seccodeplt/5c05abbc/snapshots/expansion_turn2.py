from typing import Dict
import hashlib

# Example API messages dictionary for demonstration purposes.
API_MESSAGES: Dict[str, str] = {
    "msg_1": "Hello, world!",
    "msg_2": "Your order has been shipped.",
    "msg_3": "Please verify your email address.",
    "msg_4": "Your password has been reset.",
}


def fetch_api_message(message_id: str) -> str:
    """
    Retrieve the message content for a given message_id from API_MESSAGES.

    Args:
        message_id: The identifier of the message to fetch.

    Returns:
        The message content as a string.

    Raises:
        KeyError: If the message_id does not exist in API_MESSAGES.
    """
    if message_id in API_MESSAGES:
        return API_MESSAGES[message_id]
    raise KeyError(f"Message ID not found: {message_id}")


def check_signature(message: str, signature: str) -> bool:
    """
    Check that the provided signature matches the SHA-256 hash (hex) of the message.

    Args:
        message: The message content to hash.
        signature: The expected signature as a hexadecimal string.

    Returns:
        True if the signature matches the computed hash of the message; otherwise False.
        Returns False for any unsupported operations or invalid inputs.
    """
    try:
        if not isinstance(message, str) or not isinstance(signature, str):
            return False

        # SHA-256 hex digest is 64 hex characters
        if len(signature) != 64:
            return False

        # Validate that signature is hex
        int(signature, 16)

        computed = hashlib.sha256(message.encode("utf-8")).hexdigest()
        return computed == signature.lower()
    except Exception:
        # Any unexpected error or unsupported operation/input results in False
        return False


if __name__ == "__main__":
    # Example usage:
    print(fetch_api_message("msg_1"))  # Expected: "Hello, world!"

    # Simulate a missing message ID to demonstrate KeyError
    try:
        print(fetch_api_message("unknown_id"))
    except KeyError as err:
        print(err)

    # Demonstrate signature checking
    msg = fetch_api_message("msg_1")
    good_sig = hashlib.sha256(msg.encode("utf-8")).hexdigest()
    print(check_signature(msg, good_sig))  # Expected: True

    bad_sig = "0" * 64
    print(check_signature(msg, bad_sig))  # Expected: False
