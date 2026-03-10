import hashlib
import json # For serializing data before hashing

USER_DATA = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com", "age": 30},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com", "occupation": "Engineer"},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com", "preferences": ["comics", "dogs"]},
}

def retrieve_user_profile(user_id: str) -> dict:
    """
    Retrieves the user profile for the given user_id from USER_DATA.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's profile dictionary.

    Raises:
        KeyError: If the user_id is not found in USER_DATA.
    """
    if user_id in USER_DATA:
        return USER_DATA[user_id]
    else:
        raise KeyError(f"User ID '{user_id}' not found.")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates the signature of the given data.

    Args:
        data: The data string to validate.
        signature: The signature to compare against.

    Returns:
        True if the signature is valid, False otherwise.
    """
    try:
        # It's common to hash a specific representation, e.g., UTF-8 encoded
        hashed_data = hashlib.sha256(data.encode('utf-8')).hexdigest()
        return hashed_data == signature
    except Exception:
        # Gracefully handle any errors during hashing or comparison
        return False

if __name__ == '__main__':
    # Example usage:
    try:
        profile = retrieve_user_profile("user123")
        print(f"Profile for user123: {profile}")

        profile_non_existent = retrieve_user_profile("user000")
        print(f"Profile for user000: {profile_non_existent}")
    except KeyError as e:
        print(e)

    try:
        profile_bob = retrieve_user_profile("user456")
        print(f"Profile for user456: {profile_bob}")
    except KeyError as e:
        print(e)

    # Example usage for validate_signature:
    # Simulate fetching data and a signature
    # In a real scenario, the signature would be generated and stored securely
    # when the data is created or last modified.

    # Example 1: Valid signature
    user_profile_data_str = json.dumps(USER_DATA["user123"], sort_keys=True) # Consistent serialization
    correct_signature = hashlib.sha256(user_profile_data_str.encode('utf-8')).hexdigest()
    is_valid = validate_signature(user_profile_data_str, correct_signature)
    print(f"Signature validation for user123 (correct): {is_valid}")

    # Example 2: Invalid signature
    incorrect_signature = "thisisnotthecorrectsignature"
    is_valid_incorrect = validate_signature(user_profile_data_str, incorrect_signature)
    print(f"Signature validation for user123 (incorrect): {is_valid_incorrect}")

    # Example 3: Tampered data
    tampered_data_str = json.dumps({"name": "Alice Wonderland", "email": "alice@example.com", "age": 31}, sort_keys=True) # Age changed
    is_valid_tampered = validate_signature(tampered_data_str, correct_signature)
    print(f"Signature validation for user123 (tampered data): {is_valid_tampered}")
