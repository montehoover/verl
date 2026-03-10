import hashlib

USER_DATA_STORE = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com", "age": 30},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com", "age": 45},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com", "age": 8},
}

def retrieve_user_profile(user_id: str) -> dict:
    """
    Retrieves the profile information for a given user_id from the USER_DATA_STORE.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's profile data.

    Raises:
        KeyError: If the user_id does not exist in the USER_DATA_STORE.
    """
    if user_id in USER_DATA_STORE:
        return USER_DATA_STORE[user_id]
    else:
        raise KeyError(f"User ID '{user_id}' not found.")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates if the provided signature matches the hash of the data.

    Args:
        data: The data string to hash.
        signature: The expected signature (hex digest of the hash).

    Returns:
        True if the signature matches the hash of the data, False otherwise.
    """
    try:
        # Encode the data to bytes before hashing
        data_bytes = data.encode('utf-8')
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the data bytes
        hasher.update(data_bytes)
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception as e:
        # Gracefully handle any unexpected errors during hashing or comparison
        print(f"Error during signature validation: {e}")
        return False

if __name__ == '__main__':
    # Example usage for retrieve_user_profile:
    try:
        profile = retrieve_user_profile("user123")
        print(f"Profile for user123: {profile}")
    except KeyError as e:
        print(e)

    try:
        profile = retrieve_user_profile("user000")
        print(f"Profile for user000: {profile}")
    except KeyError as e:
        print(e)

    # Example usage for validate_signature:
    sample_data = "This is some important data."
    # Pre-calculate a signature for testing (e.g., using python -c "import hashlib; print(hashlib.sha256('This is some important data.'.encode('utf-8')).hexdigest())")
    # The hash for "This is some important data." is "01a70ef3dbcf67006899de517019353200071165929519599870091002241081"
    # For demonstration, let's create a valid and an invalid signature.
    # Correct signature for sample_data
    correct_signature = hashlib.sha256(sample_data.encode('utf-8')).hexdigest()
    invalid_signature = "thisisnotthecorrectsignature"

    print(f"\nValidating signature for: '{sample_data}'")
    is_valid = validate_signature(sample_data, correct_signature)
    print(f"Signature validation result (correct signature): {is_valid}")

    is_valid = validate_signature(sample_data, invalid_signature)
    print(f"Signature validation result (invalid signature): {is_valid}")

    # Example with different data
    another_data = "Some other data."
    is_valid = validate_signature(another_data, correct_signature)
    print(f"Signature validation result (different data, original signature): {is_valid}")
