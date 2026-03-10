from typing import Dict
import hashlib

# Example API messages dictionary for demonstration purposes.
API_MESSAGES: Dict[str, str] = {
    "msg_1": "Hello, world!",
    "msg_2": "Your order has been shipped.",
    "msg_3": "Please verify your email address.",
    "msg_4": "Your password has been reset.",
}

# Predefined valid signatures for various API messages (for demonstration/context).
VALID_API_MESSAGES: Dict[str, str] = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
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


def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Fetch a message by its ID and verify its cryptographic signature.

    Args:
        msg_id: The message identifier to fetch from the API.
        sign: The signature to verify against the fetched message.

    Returns:
        True if the signature is valid for the fetched message, otherwise False.
        Returns False for any unsupported operations or invalid inputs.
    """
    try:
        if not isinstance(msg_id, str) or not isinstance(sign, str):
            return False

        message = fetch_api_message(msg_id)
        return check_signature(message, sign)
    except Exception:
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

    # Demonstrate verify_api_message
    print(verify_api_message("msg_1", good_sig))  # Expected: True
    print(verify_api_message("msg_1", bad_sig))   # Expected: False
