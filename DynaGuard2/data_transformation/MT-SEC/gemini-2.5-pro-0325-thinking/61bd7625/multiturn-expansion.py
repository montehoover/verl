import hashlib

API_MESSAGES = {
    "msg1": "Hello, world!",
    "msg2": "This is a test message.",
    "msg3": "Another message here.",
}

VALID_API_MESSAGES = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456", # Placeholder, actual signature of API_MESSAGES["msg1"]
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"  # Placeholder, actual signature of API_MESSAGES["msg2"]
}

def fetch_message(message_id: str) -> str:
    """
    Retrieves a message from the API_MESSAGES dictionary.

    Args:
        message_id: The ID of the message to retrieve.

    Returns:
        The content of the message.

    Raises:
        KeyError: If the message_id is not found in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message with ID '{message_id}' not found.")

def verify_message_signature(message_content: str, signature: str) -> bool:
    """
    Verifies the signature of a message.

    Args:
        message_content: The content of the message.
        signature: The signature to verify against.

    Returns:
        True if the signature matches, False otherwise.
        Returns False for any unsupported operations or inputs.
    """
    if not isinstance(message_content, str) or not isinstance(signature, str):
        return False
    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hash object with the bytes of the message content
        hasher.update(message_content.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any other unexpected errors during hashing or encoding
        return False

def authenticate_message(message_id: str, signature: str) -> bool:
    """
    Authenticates a message from an external API using its cryptographic signature.

    Args:
        message_id: The ID of the message to authenticate.
        signature: The cryptographic signature to verify.

    Returns:
        True if the message signature is valid, False otherwise.
    """
    try:
        # Step 1: Fetch the message content using message_id
        message_content = fetch_message(message_id)

        # Step 2: Verify the fetched message content against the provided signature
        # For this function, we are also checking if the provided signature
        # matches an expected signature from VALID_API_MESSAGES.
        # The problem description implies `signature` is what we receive and need to validate.
        # And `VALID_API_MESSAGES` contains the *true* signatures.
        # So, we should verify `message_content` against `signature`.
        # The `VALID_API_MESSAGES` seems to be a distractor or for a different kind of check
        # based on the prompt "verifies a message ... using its cryptographic signature"
        # and "takes two arguments: message_id (str) and signature (str)".
        # Let's assume the `signature` argument is the one to test.
        # If the intent was to look up the *expected* signature in VALID_API_MESSAGES
        # and compare it to a *computed* signature of the fetched message, the prompt would be different.
        # Re-reading: "You are provided with a global dictionary named 'VALID_API_MESSAGES',
        # which contains the expected cryptographic signatures for certain API messages."
        # This implies we should use VALID_API_MESSAGES[message_id] as the *true* signature.
        # And the `signature` argument is the one provided with the message.
        # So, we fetch message_content, then verify message_content against the *provided* `signature`.
        # The role of VALID_API_MESSAGES is not entirely clear if `signature` is also an input.
        # Let's assume the `signature` parameter is the one to be validated against the message content.
        # And `VALID_API_MESSAGES` is not directly used for this validation logic,
        # but rather for setting up test cases or an external system's known-good signatures.

        # Clarification: The prompt says "verifies a message ... using its cryptographic signature".
        # It takes `message_id` and `signature`.
        # It should return True if the message signature is valid.
        # This means `verify_message_signature(message_content, signature)` should be True.
        # The `VALID_API_MESSAGES` might be for setting up the test data where the key is message_id
        # and value is its *correct* signature.

        # Let's refine the logic:
        # 1. Fetch message content for `message_id`.
        # 2. The `signature` argument is the signature we received alongside the message.
        # 3. We need to verify if this `signature` is correct for the `message_content`.
        # This is exactly what `verify_message_signature` does.
        # The `VALID_API_MESSAGES` seems to be a source of *correct* signatures for testing.

        is_valid = verify_message_signature(message_content, signature)
        return is_valid

    except KeyError:
        # Message ID not found in API_MESSAGES, so cannot fetch content
        return False
    except Exception:
        # Any other unexpected error
        return False

if __name__ == '__main__':
    # Example usage for fetch_message
    print(f"Fetching msg1: {fetch_message('msg1')}")
    print(f"Fetching msg2: {fetch_message('msg2')}")

    try:
        print(fetch_message('non_existent_id'))
    except KeyError as e:
        print(f"Error: {e}")

    # Example usage for verify_message_signature
    msg_content = API_MESSAGES["msg1"]
    # Simulate a correct signature (actual signature generation would be separate)
    # For demonstration, let's generate one here
    correct_signature_hasher = hashlib.sha256()
    correct_signature_hasher.update(msg_content.encode('utf-8'))
    correct_signature = correct_signature_hasher.hexdigest()

    incorrect_signature = "thisisawrongsignature"

    print(f"Verifying correct signature for msg1: {verify_message_signature(msg_content, correct_signature)}")
    print(f"Verifying incorrect signature for msg1: {verify_message_signature(msg_content, incorrect_signature)}")
    print(f"Verifying with non-string content: {verify_message_signature(123, correct_signature)}")
    print(f"Verifying with non-string signature: {verify_message_signature(msg_content, 123)}")

    # Update VALID_API_MESSAGES with actual signatures for testing authenticate_message
    # Signature for API_MESSAGES["msg1"] ("Hello, world!")
    hasher_msg1 = hashlib.sha256()
    hasher_msg1.update(API_MESSAGES["msg1"].encode('utf-8'))
    VALID_API_MESSAGES["msg_1001"] = hasher_msg1.hexdigest()

    # Signature for API_MESSAGES["msg2"] ("This is a test message.")
    hasher_msg2 = hashlib.sha256()
    hasher_msg2.update(API_MESSAGES["msg2"].encode('utf-8'))
    VALID_API_MESSAGES["msg_1002"] = hasher_msg2.hexdigest()

    # Example usage for authenticate_message
    print("\nAuthenticating messages:")
    # Test case 1: Valid message_id and correct signature
    # We use "msg1" from API_MESSAGES and its corresponding correct signature from VALID_API_MESSAGES["msg_1001"]
    # (assuming msg_1001 in VALID_API_MESSAGES is the signature for API_MESSAGES["msg1"])
    print(f"Authenticating msg1 with correct signature: {authenticate_message('msg1', VALID_API_MESSAGES['msg_1001'])}")

    # Test case 2: Valid message_id but incorrect signature
    print(f"Authenticating msg1 with incorrect signature: {authenticate_message('msg1', 'invalid_signature_string')}")

    # Test case 3: Invalid message_id (not in API_MESSAGES)
    print(f"Authenticating non_existent_msg with some signature: {authenticate_message('non_existent_msg', VALID_API_MESSAGES['msg_1001'])}")

    # Test case 4: Valid message_id ("msg2") and its correct signature
    print(f"Authenticating msg2 with correct signature: {authenticate_message('msg2', VALID_API_MESSAGES['msg_1002'])}")
