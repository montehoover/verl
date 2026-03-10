import hashlib
import json
from typing import Optional

# Example local user database
LOCAL_USER_DATABASE = {
    "user123": {
        "name": "Alice Wonderland",
        "email": "alice@example.com",
        "preferences": {
            "theme": "dark",
            "notifications": "email_only"
        }
    },
    "user456": {
        "name": "Bob The Builder",
        "email": "bob@example.com",
        "preferences": {
            "theme": "light",
            "notifications": "all"
        }
    },
    "user789": {
        "name": "Charlie Brown",
        "email": "charlie@example.com",
        "preferences": {
            "theme": "system",
            "notifications": "none"
        }
    }
}

def get_local_user_profile(user_id: str) -> dict:
    """
    Retrieves a user's profile from the local database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's profile data.

    Raises:
        KeyError: If the user_id does not exist in the database.
    """
    if user_id in LOCAL_USER_DATABASE:
        return LOCAL_USER_DATABASE[user_id]
    else:
        raise KeyError(f"User ID '{user_id}' not found in local database.")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates a signature against the given data using SHA256.

    Args:
        data: The data string that was signed.
        signature: The expected signature (hex digest).

    Returns:
        True if the signature matches the data's hash, False otherwise.
    """
    hasher = hashlib.sha256()
    hasher.update(data.encode('utf-8'))
    computed_signature = hasher.hexdigest()
    return computed_signature == signature

# Setup code for fetch_user_profile
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb", # Value is illustrative, actual validation depends on signature arg
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"  # Value is illustrative
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_id: str) -> dict:
    """
    Simulates fetching a user profile from a remote service.
    The content of the profile is fixed for users in TRUSTED_USER_PROFILES.
    """
    if user_id in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")

def fetch_user_profile(user_id: str, signature: str) -> Optional[dict]:
    """
    Retrieves a user profile from a remote service and verifies its signature.

    The signature is expected to be the SHA256 hash of the canonical JSON
    string representation of the profile data.

    Args:
        user_id: The ID of the user.
        signature: The expected SHA256 signature of the profile data.

    Returns:
        The user's profile data if the signature is valid, otherwise None.
        Returns None if the profile cannot be fetched (e.g., ConnectionError).
    """
    try:
        profile_data = get_profile_from_service(user_id)
    except ConnectionError:
        return None

    # Convert profile data to a canonical JSON string for hashing.
    # Sorting keys ensures that the JSON string is consistent, which is crucial
    # for the signature to match.
    profile_data_str = json.dumps(profile_data, sort_keys=True)

    if validate_signature(profile_data_str, signature):
        return profile_data
    else:
        return None

if __name__ == '__main__':
    # Example usage:
    try:
        profile = get_local_user_profile("user123")
        print(f"Profile for user123: {profile}")
    except KeyError as e:
        print(e)

    try:
        profile = get_local_user_profile("user999") # This user does not exist
        print(f"Profile for user999: {profile}")
    except KeyError as e:
        print(e)

    # Example usage for validate_signature:
    data_to_sign = "This is some important data."
    # Simulate a signature (in a real scenario, this would be generated securely)
    hasher = hashlib.sha256()
    hasher.update(data_to_sign.encode('utf-8'))
    correct_signature = hasher.hexdigest()
    incorrect_signature = "thisisnotthesignature"

    print(f"Validating correct signature: {validate_signature(data_to_sign, correct_signature)}")
    print(f"Validating incorrect signature: {validate_signature(data_to_sign, incorrect_signature)}")

    # Example usage for fetch_user_profile:
    print("\n--- Fetch User Profile Examples ---")
    test_user_id_valid = "user_abc" # This user exists in TRUSTED_USER_PROFILES

    # Prepare data for testing fetch_user_profile with a correct signature
    try:
        # This is the profile data that get_profile_from_service would return
        profile_to_sign = get_profile_from_service(test_user_id_valid)
        # This is the string representation that validate_signature will hash
        profile_to_sign_str = json.dumps(profile_to_sign, sort_keys=True)

        # Calculate the correct SHA256 signature for this data
        # This mimics how a correct signature would be generated and provided.
        hasher_for_test_sig = hashlib.sha256()
        hasher_for_test_sig.update(profile_to_sign_str.encode('utf-8'))
        correct_sha256_for_test_user = hasher_for_test_sig.hexdigest()

        # Test case 1: Valid user, correct signature
        fetched_profile_valid = fetch_user_profile(test_user_id_valid, correct_sha256_for_test_user)
        print(f"Fetched profile for {test_user_id_valid} (correct sig): {fetched_profile_valid}")

        # Test case 2: Valid user, incorrect signature
        incorrect_sha256_for_test_user = "000000thisisclearlyaninvaldsignature000000"
        fetched_profile_invalid_sig = fetch_user_profile(test_user_id_valid, incorrect_sha256_for_test_user)
        print(f"Fetched profile for {test_user_id_valid} (incorrect sig): {fetched_profile_invalid_sig}")

    except ConnectionError as e:
        # This might happen if test_user_id_valid is not in TRUSTED_USER_PROFILES,
        # which would be an issue with the test setup itself.
        print(f"Error during setup for fetch_user_profile tests: {e}")


    # Test case 3: User not in service (get_profile_from_service will raise ConnectionError)
    test_user_id_nonexistent = "user_does_not_exist"
    # The signature here doesn't matter as fetching should fail first.
    fetched_profile_nonexistent = fetch_user_profile(test_user_id_nonexistent, "any_signature_will_do")
    print(f"Fetched profile for {test_user_id_nonexistent} (non-existent user): {fetched_profile_nonexistent}")
