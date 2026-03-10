"""
Utilities for working with local messages and API message verification.
"""
import hmac

# Predefined local messages dictionary with some example message IDs
LOCAL_MESSAGES = {
    "msg-001": {"text": "Hello there"},
    "msg-002": {"text": "Welcome back"},
    "msg-003": {"text": "Goodbye"},
    "abc123": {"text": "Sample message"},
    "xyz789": {"text": "Another sample"},
}

# Expected cryptographic signatures for different API messages
VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def check_message_existence(message_id: str, expected_value: str) -> bool:
    """
    Check if a given message ID exists in the LOCAL_MESSAGES dictionary and
    verify its stored value matches the expected value.

    The stored value is the 'text' field when the message entry is a dict.
    If the entry is a plain string, it will be compared directly.

    Args:
        message_id (str): The message ID to check.
        expected_value (str): The expected value to compare against.

    Returns:
        bool: True if the message ID exists and matches the expected value, False otherwise.
    """
    record = LOCAL_MESSAGES.get(message_id)
    if record is None:
        return False

    if isinstance(record, dict):
        return record.get("text") == expected_value

    if isinstance(record, str):
        return record == expected_value

    return False


def verify_message(api_message_id: str, crypto_signature: str) -> bool:
    """
    Verify the legitimacy of an API message by checking its cryptographic signature.

    Args:
        api_message_id (str): Unique identifier for the API message.
        crypto_signature (str): The cryptographic signature to validate.

    Returns:
        bool: True if the signature is valid for the given api_message_id, False otherwise.

    Raises:
        KeyError: If the given api_message_id does not exist in VALID_API_MESSAGES.
    """
    expected_signature = VALID_API_MESSAGES[api_message_id]  # May raise KeyError
    return hmac.compare_digest(expected_signature, crypto_signature)
