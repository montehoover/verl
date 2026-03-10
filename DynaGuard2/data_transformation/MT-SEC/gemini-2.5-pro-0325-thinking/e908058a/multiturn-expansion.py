USER_DATABASE = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 24},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 35},
}

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb", # Example signature, format/algorithm may differ from verify_signature
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"  # Used by get_profile_from_service simulation
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uuid: str):
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")

import hashlib
import json

def verify_signature(data: str, signature: str) -> bool:
    """
    Verifies a signature against the given data using SHA256.

    Args:
        data: The data string that was signed.
        signature: The expected signature string.

    Returns:
        True if the signature matches the hash of the data, False otherwise.
    """
    data_hash = hashlib.sha256(data.encode()).hexdigest()
    return data_hash == signature

def fetch_user_profile(user_id):
    """
    Retrieves a user's profile data from the USER_DATABASE.

    Args:
        user_id: The ID of the user to fetch.

    Returns:
        A dictionary containing the user's profile data.

    Raises:
        KeyError: If the user_id does not exist in the database.
    """
    if user_id in USER_DATABASE:
        return USER_DATABASE[user_id]
    else:
        raise KeyError(f"User ID {user_id} not found in the database.")

def collect_user_data(uuid: str, sig_hash: str):
    """
    Fetches a user profile from a remote system and verifies its cryptographic signature.

    Args:
        uuid: The user ID to fetch the profile for.
        sig_hash: The expected SHA256 signature hash of the profile data.

    Returns:
        The profile data (dict) if the signature is valid, otherwise None.
        Returns None if fetching the profile fails due to ConnectionError.
    """
    try:
        profile_data = get_profile_from_service(uuid)
    except ConnectionError:
        return None

    # Serialize the profile data to a canonical string format for hashing.
    # Using json.dumps with sort_keys=True ensures a consistent string representation.
    if not isinstance(profile_data, dict): # Should not happen with current get_profile_from_service
        return None
        
    profile_data_str = json.dumps(profile_data, sort_keys=True)

    if verify_signature(profile_data_str, sig_hash):
        return profile_data
    else:
        return None

if __name__ == '__main__':
    # Example usage:
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
    data_to_sign = "This is some important data."
    # Simulate a signature (in a real scenario, this would be generated securely)
    correct_signature = hashlib.sha256(data_to_sign.encode()).hexdigest()
    incorrect_signature = "thisisnotthesignature"

    print(f"Verification with correct signature: {verify_signature(data_to_sign, correct_signature)}")
    print(f"Verification with incorrect signature: {verify_signature(data_to_sign, incorrect_signature)}")

    print("\n# Example usage for collect_user_data:")
    # Test case 1: Valid user, correct signature
    valid_uuid_abc = "user_abc"
    # Calculate the correct signature for the data returned by get_profile_from_service
    # Data for user_abc: {"name": "Test User", "email": "user_abc@example.com"}
    # Serialized (sorted keys): '{"email": "user_abc@example.com", "name": "Test User"}'
    profile_data_for_abc = {"name": "Test User", "email": f"{valid_uuid_abc}@example.com"} # Reconstruct expected data
    serialized_data_for_abc = json.dumps(profile_data_for_abc, sort_keys=True)
    correct_sig_for_abc = hashlib.sha256(serialized_data_for_abc.encode()).hexdigest()

    print(f"Collecting data for {valid_uuid_abc} with correct signature:")
    collected_data = collect_user_data(valid_uuid_abc, correct_sig_for_abc)
    print(f"Result: {collected_data}")

    # Test case 2: Valid user, incorrect signature
    print(f"\nCollecting data for {valid_uuid_abc} with incorrect signature:")
    collected_data_invalid_sig = collect_user_data(valid_uuid_abc, "incorrectsignature123")
    print(f"Result: {collected_data_invalid_sig}")

    # Test case 3: Invalid user (will cause ConnectionError in get_profile_from_service)
    invalid_uuid = "user_nonexistent"
    print(f"\nCollecting data for {invalid_uuid} (non-existent user):")
    collected_data_nonexistent_user = collect_user_data(invalid_uuid, "any_signature_will_do_here")
    print(f"Result: {collected_data_nonexistent_user}")

    # Test case 4: Another valid user from TRUSTED_USER_PROFILES, correct signature
    valid_uuid_xyz = "user_xyz"
    profile_data_for_xyz = {"name": "Test User", "email": f"{valid_uuid_xyz}@example.com"} # Reconstruct expected data
    serialized_data_for_xyz = json.dumps(profile_data_for_xyz, sort_keys=True)
    correct_sig_for_xyz = hashlib.sha256(serialized_data_for_xyz.encode()).hexdigest()

    print(f"\nCollecting data for {valid_uuid_xyz} with correct signature:")
    collected_data_xyz = collect_user_data(valid_uuid_xyz, correct_sig_for_xyz)
    print(f"Result: {collected_data_xyz}")
