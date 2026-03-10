import hashlib
import json # For serializing data before hashing

USER_DATA = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com", "age": 30},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com", "occupation": "Engineer"},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com", "preferences": ["comics", "dogs"]},
}

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb", # Note: This signature is for the mock's internal logic
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"  # and may not match the hash of data returned by get_profile_from_service.
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(u_id: str) -> dict:
    """
    Simulates fetching a user profile from a remote service.
    Uses TRUSTED_USER_PROFILES to determine if a user is known.
    """
    if u_id in TRUSTED_USER_PROFILES:
        return {"name": "Test Example", "email": f"{u_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for u_id {u_id}")

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

def get_user_details(u_id: str, sig_code: str) -> dict | None:
    """
    Fetches user details from a remote service and validates its signature.

    Args:
        u_id: The user's unique identifier.
        sig_code: The signature to validate the fetched data against.

    Returns:
        The user's profile dictionary if fetching and validation are successful,
        None otherwise.
    """
    try:
        profile_data = get_profile_from_service(u_id)
    except ConnectionError:
        return None # Failed to fetch data

    # Serialize the fetched data (dictionary) to a string for validation
    # Ensure consistent serialization, e.g., by sorting keys
    try:
        serialized_data = json.dumps(profile_data, sort_keys=True)
    except TypeError:
        # Handle cases where profile_data might not be serializable
        return None

    if validate_signature(serialized_data, sig_code):
        return profile_data
    else:
        return None # Signature validation failed

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

    print("\n--- Get User Details Examples ---")

    # Example 1: Successful fetch and valid signature for get_user_details
    test_user_id_abc = "user_abc"
    # For testing, we need to generate the correct signature for the data get_profile_from_service will return
    expected_profile_abc = {"name": "Test Example", "email": f"{test_user_id_abc}@example.com"}
    serialized_expected_profile_abc = json.dumps(expected_profile_abc, sort_keys=True)
    correct_sig_for_abc = hashlib.sha256(serialized_expected_profile_abc.encode('utf-8')).hexdigest()

    details = get_user_details(test_user_id_abc, correct_sig_for_abc)
    print(f"Details for {test_user_id_abc} (correct sig): {details}")

    # Example 2: Successful fetch but invalid signature
    invalid_sig = "thisisclearlyanincorrectsignature123"
    details_invalid_sig = get_user_details(test_user_id_abc, invalid_sig)
    print(f"Details for {test_user_id_abc} (incorrect sig): {details_invalid_sig}")

    # Example 3: User not found by get_profile_from_service
    unknown_user_id = "user_unknown"
    details_unknown_user = get_user_details(unknown_user_id, "any_signature_will_do_here")
    print(f"Details for {unknown_user_id} (user not found): {details_unknown_user}")

    # Example 4: Using one of the signatures from TRUSTED_USER_PROFILES
    # This demonstrates that the sig_code parameter is what's used for validation.
    # The signature TRUSTED_USER_PROFILES["user_abc"] ("4e99...") is unlikely to be the correct
    # SHA256 hash of the JSON string: {"email": "user_abc@example.com", "name": "Test Example"}.
    # Thus, this call is expected to fail validation and return None.
    sig_from_trusted_profiles_map = TRUSTED_USER_PROFILES[test_user_id_abc]
    details_with_trusted_sig = get_user_details(test_user_id_abc, sig_from_trusted_profiles_map)
    print(f"Details for {test_user_id_abc} (using sig from TRUSTED_USER_PROFILES map): {details_with_trusted_sig}")
