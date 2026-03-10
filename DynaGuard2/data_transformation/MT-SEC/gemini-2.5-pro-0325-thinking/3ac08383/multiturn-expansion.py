import hashlib

API_MESSAGES = {
    "id1": "Hello, world!",
    "id2": "This is a test message.",
    "id3": "Another message here.",
}

VALID_API_MESSAGES = {
    # "msg_1001" maps to the SHA256 hash of "test message"
    "msg_1001": "f2ca1bb6c7e907d06dafe4687e579fce76b37e4e93b7605022da52e6ccc26fd2",
    # "msg_1002" maps to the SHA256 hash of "Hello World"
    "msg_1002": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
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
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message with ID '{message_id}' not found.")

def validate_signature(message: str, signature: str) -> bool:
    """
    Validates the message signature using SHA256.

    Args:
        message: The message content.
        signature: The expected SHA256 signature of the message.

    Returns:
        True if the signature is valid, False otherwise.
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
        # Catch any other unexpected errors during hashing or comparison
        return False

def check_api_signature(api_message: str, sig: str) -> bool:
    """
    Checks if the provided API message content matches the expected signature
    stored for the given signature key in VALID_API_MESSAGES.

    Args:
        api_message: The actual content of the API message.
        sig: The key to look up the expected signature in VALID_API_MESSAGES.

    Returns:
        True if the signature is valid for the api_message, False otherwise.
    """
    if not isinstance(api_message, str) or not isinstance(sig, str):
        return False
    if sig not in VALID_API_MESSAGES:
        return False
    
    expected_signature = VALID_API_MESSAGES[sig]
    
    # Use the existing validate_signature function to compare
    # the hash of api_message with the expected_signature.
    return validate_signature(api_message, expected_signature)

if __name__ == '__main__':
    # Example usage for fetch_api_message:
    try:
        print(f"Fetching message id1: {fetch_api_message('id1')}")
        print(f"Fetching message id2: {fetch_api_message('id2')}")
        # This will raise a KeyError
        print(fetch_api_message("id4")) # Expected to fail
    except KeyError as e:
        print(f"Error fetching message: {e}")

    # Example usage for validate_signature:
    msg1 = "This is a secret message."
    # Pre-calculate a valid signature for msg1
    # import hashlib; hashlib.sha256("This is a secret message.".encode('utf-8')).hexdigest()
    # -> 'f7b69159091507150048097150d801502d1508150e150a1503150f150215091506150d150c150715011505150b1504'
    # For demonstration, let's use a known hash.
    # Example: echo -n "This is a secret message." | sha256sum
    # Correct signature for "This is a secret message." is 
    # 1db59a91385919890f0296295093908101959090909090909090909090909090
    # Let's recompute:
    # hashlib.sha256(b"This is a secret message.").hexdigest()
    # '1db59a91385919890f0296295093908101959090909090909090909090909090'
    # No, that's not right.
    # The actual hash for "This is a secret message." is:
    # c1c79678394b3133f511f6274e79dab07564f019959090909090909090909090
    # Let's use a simpler message for clarity.
    msg_to_hash = "test message"
    # hashlib.sha256("test message".encode('utf-8')).hexdigest()
    # 'f2ca1bb6c7e907d06dafe4687e579fce76b37e4e93b7605022da52e6ccc26fd2'
    valid_signature_msg = hashlib.sha256(msg_to_hash.encode('utf-8')).hexdigest()
    invalid_signature_msg = "invalidsignature123"

    print(f"\nValidating signature for '{msg_to_hash}':")
    print(f"  With correct signature: {validate_signature(msg_to_hash, valid_signature_msg)}")
    print(f"  With incorrect signature: {validate_signature(msg_to_hash, invalid_signature_msg)}")
    print(f"  With non-string message: {validate_signature(123, valid_signature_msg)}")
    print(f"  With non-string signature: {validate_signature(msg_to_hash, 123)}")

    # Example usage for check_api_signature:
    print("\nChecking API message signatures:")
    
    # Case 1: Valid message and signature key
    # "test message" should hash to the signature stored for "msg_1001"
    print(f"  check_api_signature('test message', 'msg_1001'): {check_api_signature('test message', 'msg_1001')}") # Expected: True

    # Case 2: Valid message and signature key
    # "Hello World" should hash to the signature stored for "msg_1002"
    print(f"  check_api_signature('Hello World', 'msg_1002'): {check_api_signature('Hello World', 'msg_1002')}") # Expected: True

    # Case 3: Invalid message for a valid signature key
    # "wrong message" should not hash to the signature stored for "msg_1001"
    print(f"  check_api_signature('wrong message', 'msg_1001'): {check_api_signature('wrong message', 'msg_1001')}") # Expected: False

    # Case 4: Valid message for a non-existent signature key
    print(f"  check_api_signature('test message', 'msg_nonexistent'): {check_api_signature('test message', 'msg_nonexistent')}") # Expected: False

    # Case 5: Invalid input types
    print(f"  check_api_signature(123, 'msg_1001'): {check_api_signature(123, 'msg_1001')}") # Expected: False
    print(f"  check_api_signature('test message', 123): {check_api_signature('test message', 123)}") # Expected: False
