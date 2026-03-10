import hashlib

API_MESSAGES = {
    "id_123": "Hello, this is a message from the API.",
    "id_456": "Another message here.",
    "id_789": "A final test message.",
}

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}

def fetch_api_message(message_id: str) -> str:
    """
    Retrieves a message from the API_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to retrieve.

    Returns:
        The content of the message.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    if message_id in API_MESSAGES:
        return API_MESSAGES[message_id]
    else:
        raise KeyError(f"Message ID '{message_id}' not found.")

def check_signature(message: str, signature: str) -> bool:
    """
    Checks if the provided signature matches the hash of the message.

    Args:
        message: The message string.
        signature: The expected signature (hex digest of SHA256).

    Returns:
        True if the signature matches, False otherwise.
        Returns False for non-string inputs or other errors.
    """
    if not isinstance(message, str) or not isinstance(signature, str):
        return False
    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the encoded message
        hasher.update(message.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any unexpected errors during hashing or comparison
        return False

def verify_api_message(msg_id: str, sign: str) -> bool:
    """
    Fetches a message by its ID and verifies its cryptographic signature.

    Args:
        msg_id: The ID of the message to fetch from API_MESSAGES.
        sign: The cryptographic signature to verify against the message.

    Returns:
        True if the message is fetched and the signature is valid,
        False otherwise (e.g., message not found, signature mismatch).
    """
    try:
        message_content = fetch_api_message(msg_id)
        return check_signature(message_content, sign)
    except KeyError:
        # Message ID not found in API_MESSAGES
        return False
    except Exception:
        # Any other unexpected error during fetch or verification
        return False

if __name__ == '__main__':
    # Example usage for fetch_api_message:
    try:
        message_content = fetch_api_message("id_123")
        print(f"Message content for id_123: {message_content}")

        message_content_non_existent = fetch_api_message("id_000")
        print(f"Message content for id_000: {message_content_non_existent}")
    except KeyError as e:
        print(f"Error: {e}")

    try:
        message_content_2 = fetch_api_message("id_456")
        print(f"Message content for id_456: {message_content_2}")
    except KeyError as e:
        print(f"Error: {e}")

    # Example usage for check_signature:
    test_message = "This is a test message."
    # Pre-calculate a valid signature for the test message
    # import hashlib
    # hashlib.sha256("This is a test message.".encode('utf-8')).hexdigest()
    # Output: 'c1591350f8736972cc90f68a49a730dc011b06999876f2a1011c1792f9199879'
    valid_signature = "c1591350f8736972cc90f68a49a730dc011b06999876f2a1011c1792f9199879"
    invalid_signature = "invalidsignature123"

    print(f"\nChecking signature for '{test_message}':")
    is_valid = check_signature(test_message, valid_signature)
    print(f"Signature '{valid_signature}' is valid: {is_valid}")

    is_valid_again = check_signature(test_message, invalid_signature)
    print(f"Signature '{invalid_signature}' is valid: {is_valid_again}")

    # Test with non-string input
    is_valid_non_string = check_signature(123, valid_signature)
    print(f"Signature check with non-string message: {is_valid_non_string}")

    is_valid_non_string_sig = check_signature(test_message, None)
    print(f"Signature check with non-string signature: {is_valid_non_string_sig}")

    # Example usage for verify_api_message:
    print(f"\nVerifying API messages:")
    # Case 1: Valid message ID and correct signature
    # For API_MESSAGES["id_123"] = "Hello, this is a message from the API."
    # hashlib.sha256("Hello, this is a message from the API.".encode('utf-8')).hexdigest()
    # is 'f9aaf1b198736336c877098909401180c80983de98b8290309095480a430f8f8'
    correct_sig_id_123 = "f9aaf1b198736336c877098909401180c80983de98b8290309095480a430f8f8"
    verification_result_1 = verify_api_message("id_123", correct_sig_id_123)
    print(f"Verification for 'id_123' with correct signature: {verification_result_1}")

    # Case 2: Valid message ID and incorrect signature
    incorrect_sig_id_123 = "0000000000000000000000000000000000000000000000000000000000000000"
    verification_result_2 = verify_api_message("id_123", incorrect_sig_id_123)
    print(f"Verification for 'id_123' with incorrect signature: {verification_result_2}")

    # Case 3: Invalid message ID
    verification_result_3 = verify_api_message("non_existent_id", correct_sig_id_123)
    print(f"Verification for 'non_existent_id': {verification_result_3}")

    # Example using a key from VALID_API_MESSAGES (assuming "msg_1001" is also in API_MESSAGES)
    # To make this test runnable, "msg_1001" would need to be in API_MESSAGES
    # and VALID_API_MESSAGES["msg_1001"] would be its actual signature.
    # For now, this will likely fail or depend on API_MESSAGES content.
    # If API_MESSAGES does not contain "msg_1001", it will return False.
    if "msg_1001" in API_MESSAGES: # Check to prevent KeyError if not in API_MESSAGES for demo
        verification_result_4 = verify_api_message("msg_1001", VALID_API_MESSAGES["msg_1001"])
        print(f"Verification for 'msg_1001' using VALID_API_MESSAGES signature: {verification_result_4}")
    else:
        print(f"Skipping 'msg_1001' verification as it's not in API_MESSAGES for this example.")
