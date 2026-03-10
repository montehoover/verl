import hashlib
import string

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

API_MESSAGES = {
    "welcome": "Welcome to the service!",
    "farewell": "Goodbye and take care!",
    "status_ok": "Your request was successful.",
    "info_42": "The answer is 42.",
    "msg_1001": "External API message 1001",
    "msg_1002": "External API message 1002"
}

def fetch_api_message(message_id):
    """
    Retrieve a message from the API_MESSAGES dictionary by its ID.

    Args:
        message_id: The identifier for the message to fetch.

    Returns:
        The message content associated with the given message_id.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError as e:
        raise KeyError(f"Message ID '{message_id}' not found in API_MESSAGES.") from e

def validate_signature(message, signature):
    """
    Validate that the provided signature matches the SHA-256 hash of the message.

    Args:
        message (str): The message to validate.
        signature (str): The expected SHA-256 hex digest of the message.

    Returns:
        bool: True if the signature matches the computed hash, False otherwise.
    """
    try:
        if not isinstance(message, str) or not isinstance(signature, str):
            return False

        sig = signature.strip()
        # Expect a 64-character hex string for SHA-256
        if len(sig) != 64 or any(c not in string.hexdigits for c in sig):
            return False

        computed = hashlib.sha256(message.encode("utf-8")).hexdigest()
        return computed == sig.lower()
    except Exception:
        # Any unsupported operation or unexpected input should result in False
        return False

def check_api_signature(api_message, sig):
    """
    Integrate message retrieval and signature verification.
    Verifies that the provided signature matches the expected signature for the given API message.

    Args:
        api_message (str): The identifier of the API message to fetch and verify.
        sig (str): The signature to verify against the expected one.

    Returns:
        bool: True if the signature is valid for the given api_message, False otherwise.
    """
    try:
        if not isinstance(api_message, str) or not isinstance(sig, str):
            return False

        # Attempt to fetch the message to ensure the API message exists.
        # If the message does not exist, this will raise a KeyError and we return False.
        _ = fetch_api_message(api_message)

        # Compare against the expected signature stored for this message ID.
        expected = VALID_API_MESSAGES.get(api_message)
        if expected is None:
            return False

        return sig == expected
    except Exception:
        return False


if __name__ == "__main__":
    # Demonstration of successful retrieval
    print("Existing ID:", fetch_api_message("welcome"))

    # Demonstration of missing ID causing a KeyError
    try:
        print("Missing ID:", fetch_api_message("does_not_exist"))
    except KeyError as err:
        print("Error:", err)

    # Demonstration of signature validation
    msg = "Welcome to the service!"
    good_sig = hashlib.sha256(msg.encode("utf-8")).hexdigest()
    bad_sig = "0" * 64

    print("Signature valid (expected True):", validate_signature(msg, good_sig))
    print("Signature valid (expected False):", validate_signature(msg, bad_sig))

    # Demonstration of API signature check
    print("Check API signature (expected True):", check_api_signature("msg_1001", "abcd1234efgh5678ijkl9012mnop3456"))
    print("Check API signature (expected False):", check_api_signature("msg_1001", "wrong_signature"))
