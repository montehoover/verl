import hashlib
import json

# Sample user database
USER_DATABASE = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 24},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 35},
}

def fetch_user_profile(user_id: int) -> dict:
    """
    Retrieves a user's profile from the USER_DATABASE.

    Args:
        user_id: The ID of the user to fetch.

    Returns:
        The user's profile data.

    Raises:
        KeyError: If the user_id does not exist in the database.
    """
    if user_id in USER_DATABASE:
        return USER_DATABASE[user_id]
    else:
        raise KeyError(f"User ID {user_id} not found in the database.")

def verify_signature(data: str, signature: str) -> bool:
    """
    Verifies a signature against the hash of the data.

    Args:
        data: The data that was signed.
        signature: The signature to verify.

    Returns:
        True if the signature matches the hash of the data, False otherwise.
    """
    data_hash = hashlib.sha256(data.encode()).hexdigest()
    return data_hash == signature

# Setup code for collect_user_data
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",  # Example signature (format might differ from SHA256)
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"   # Example signature
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_identifier: str) -> dict:
    """
    Simulates fetching a user profile from an external service.
    """
    if user_identifier in TRUSTED_USER_PROFILES:
        # Simulate returning profile data based on user_identifier
        return {"name": "Test User", "id": user_identifier, "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")

def collect_user_data(user_identifier: str, user_signature: str) -> dict:
    """
    Fetches a user profile from a service and verifies its cryptographic signature.

    Args:
        user_identifier: The identifier of the user.
        user_signature: The cryptographic signature of the user's profile data.

    Returns:
        The user's profile data if the signature is successfully verified.

    Raises:
        ConnectionError: If the profile cannot be fetched.
        ValueError: If the signature verification fails.
    """
    # 1. Fetch profile data from the service
    profile_data = get_profile_from_service(user_identifier)

    # 2. Serialize profile data to a canonical string format (sorted JSON)
    # This ensures that the same data always produces the same string representation
    profile_data_str = json.dumps(profile_data, sort_keys=True)

    # 3. Verify the signature
    if verify_signature(profile_data_str, user_signature):
        return profile_data
    else:
        raise ValueError(f"Signature verification failed for user {user_identifier}.")

if __name__ == '__main__':
    # Example usage for fetch_user_profile:
    try:
        user_profile = fetch_user_profile(1)
        print(f"User profile for ID 1: {user_profile}")

        user_profile_non_existent = fetch_user_profile(4)
        print(f"User profile for ID 4: {user_profile_non_existent}")
    except KeyError as e:
        print(e)

    try:
        user_profile_bob = fetch_user_profile(2)
        print(f"User profile for ID 2: {user_profile_bob}")
    except KeyError as e:
        print(e)

    # Example usage for verify_signature:
    sample_data = "This is some important data."
    # Simulate a correct signature (actual hash of sample_data)
    correct_signature = hashlib.sha256(sample_data.encode()).hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nVerifying correct signature: {verify_signature(sample_data, correct_signature)}")
    print(f"Verifying incorrect signature: {verify_signature(sample_data, incorrect_signature)}")

    another_data = "Some other data"
    another_signature = hashlib.sha256(another_data.encode()).hexdigest()
    print(f"Verifying signature for '{another_data}': {verify_signature(another_data, another_signature)}")

    # Example usage for collect_user_data:
    print("\n--- Testing collect_user_data ---")

    # Case 1: Successful data collection and verification
    user_id_success = "user_abc"
    try:
        # Simulate fetching the data to generate the correct signature for the test
        profile_to_sign = get_profile_from_service(user_id_success)
        profile_to_sign_str = json.dumps(profile_to_sign, sort_keys=True)
        correct_user_sig = hashlib.sha256(profile_to_sign_str.encode()).hexdigest()

        print(f"Attempting to collect data for {user_id_success} with correct signature...")
        collected_data = collect_user_data(user_id_success, correct_user_sig)
        print(f"Successfully collected data for {user_id_success}: {collected_data}")
    except (ConnectionError, ValueError) as e:
        print(f"Error collecting data for {user_id_success}: {e}")

    # Case 2: Signature verification failure
    user_id_fail_sig = "user_xyz"
    incorrect_user_sig = "invalidsignature12345"
    try:
        print(f"\nAttempting to collect data for {user_id_fail_sig} with incorrect signature...")
        collected_data_fail = collect_user_data(user_id_fail_sig, incorrect_user_sig)
        print(f"Collected data for {user_id_fail_sig}: {collected_data_fail}") # Should not reach here
    except (ConnectionError, ValueError) as e:
        print(f"Error collecting data for {user_id_fail_sig}: {e}")

    # Case 3: User not found (ConnectionError)
    user_id_not_found = "user_nonexistent"
    try:
        print(f"\nAttempting to collect data for {user_id_not_found}...")
        collected_data_not_found = collect_user_data(user_id_not_found, "somesignature")
        print(f"Collected data for {user_id_not_found}: {collected_data_not_found}") # Should not reach here
    except (ConnectionError, ValueError) as e:
        print(f"Error collecting data for {user_id_not_found}: {e}")
