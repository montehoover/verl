import hashlib
import json

USER_DATA = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 24},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 35},
}

# TRUSTED_USER_PROFILES stores UID mapped to the SHA256 hash of their profile data
# (profile data is JSON string with sorted keys).
# Example profile for user_abc: {"name": "Test Individual", "email": "user_abc@example.com"}
# JSON string: {"email": "user_abc@example.com", "name": "Test Individual"}
# SHA256 hash: 0875c918b202ce12b95ded89074170750a81280459772d3000f9906580bbe95
#
# Example profile for user_xyz: {"name": "Test Individual", "email": "user_xyz@example.com"}
# JSON string: {"email": "user_xyz@example.com", "name": "Test Individual"}
# SHA256 hash: 4a664002958392f2076930b191302664d10210380061590709175f83590a98cb
TRUSTED_USER_PROFILES = {
    "user_abc": "0875c918b202ce12b95ded89074170750a81280459772d3000f9906580bbe95",
    "user_xyz": "4a664002958392f2076930b191302664d10210380061590709175f83590a98cb"
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uid: str):
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        # This case should ideally not be hit if retrieve_user_profile checks uid first
        raise ConnectionError(f"Could not fetch profile for uid {uid}")

def get_user_profile(user_id):
    """
    Retrieves the user profile for a given user_id from USER_DATA.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's profile data.

    Raises:
        KeyError: If the user_id is not found in USER_DATA.
    """
    if user_id in USER_DATA:
        return USER_DATA[user_id]
    else:
        raise KeyError(f"User ID {user_id} not found.")

def check_signature(data: str, signature: str) -> bool:
    """
    Checks if the SHA256 hash of the data matches the given signature.

    Args:
        data: The data string to hash.
        signature: The expected hash signature.

    Returns:
        True if the hash of the data matches the signature, False otherwise.
    """
    data_hash = hashlib.sha256(data.encode()).hexdigest()
    return data_hash == signature

def retrieve_user_profile(uid: str, sig: str):
    """
    Retrieves a user profile from a service and verifies its cryptographic signature.

    Args:
        uid: The user ID.
        sig: The cryptographic signature provided for the user's profile data.

    Returns:
        The user's profile data if the signature is valid and matches the expected signature.

    Raises:
        KeyError: If the uid is not recognized in TRUSTED_USER_PROFILES.
        ValueError: If the provided signature does not match the trusted signature,
                    or if the fetched data does not match the trusted signature.
        ConnectionError: If fetching the profile from the service fails.
    """
    if uid not in TRUSTED_USER_PROFILES:
        raise KeyError(f"User ID '{uid}' not recognized or no trusted signature available.")

    expected_trusted_signature = TRUSTED_USER_PROFILES[uid]

    if sig != expected_trusted_signature:
        raise ValueError("Provided signature does not match the expected trusted signature for this user.")

    # At this point, 'sig' is the correct, trusted signature.
    # Now, fetch the data and verify it against this signature.
    try:
        profile_data = get_profile_from_service(uid)
    except ConnectionError as e:
        # Let ConnectionError propagate if service fails
        raise e

    # Convert profile_data dictionary to a canonical JSON string for hashing
    # Sorting keys ensures that the JSON string is consistent
    profile_data_str = json.dumps(profile_data, sort_keys=True)

    if not check_signature(profile_data_str, sig):
        # This is a critical error: the data from the service does not match its trusted signature
        raise ValueError("Data integrity check failed: The fetched profile data does not match its trusted signature.")

    return profile_data

if __name__ == '__main__':
    # Example usage:
    try:
        print(f"Profile for user 1: {get_user_profile(1)}")
        print(f"Profile for user 2: {get_user_profile(2)}")
        # This will raise a KeyError
        print(f"Profile for user 4: {get_user_profile(4)}")
    except KeyError as e:
        print(e)

    # Example usage for check_signature:
    data_to_sign = "This is some important data."
    # Simulate a signature (in a real scenario, this would be generated securely)
    correct_signature = hashlib.sha256(data_to_sign.encode()).hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nChecking signature for correct data: {check_signature(data_to_sign, correct_signature)}")
    print(f"Checking signature with incorrect signature: {check_signature(data_to_sign, incorrect_signature)}")
    print(f"Checking signature for different data: {check_signature('Some other data.', correct_signature)}")

    print("\n--- Testing retrieve_user_profile ---")
    # Scenario 1: Successful retrieval
    uid_good = "user_abc"
    sig_good = TRUSTED_USER_PROFILES[uid_good]
    try:
        profile = retrieve_user_profile(uid_good, sig_good)
        print(f"Successfully retrieved profile for {uid_good}: {profile}")
    except (KeyError, ValueError, ConnectionError) as e:
        print(f"Error for {uid_good} (good sig): {e}")

    # Scenario 2: Incorrect signature provided
    uid_bad_sig_test = "user_abc"
    sig_bad = "incorrectsignature12345"
    try:
        profile = retrieve_user_profile(uid_bad_sig_test, sig_bad)
        print(f"Successfully retrieved profile for {uid_bad_sig_test} (bad sig): {profile}")
    except (KeyError, ValueError, ConnectionError) as e:
        print(f"Error for {uid_bad_sig_test} (bad sig): {e}")

    # Scenario 3: UID not recognized
    uid_unknown = "user_unknown"
    sig_for_unknown = "any_signature"
    try:
        profile = retrieve_user_profile(uid_unknown, sig_for_unknown)
        print(f"Successfully retrieved profile for {uid_unknown}: {profile}")
    except (KeyError, ValueError, ConnectionError) as e:
        print(f"Error for {uid_unknown}: {e}")

    # Scenario 4: Trusted signature on record does not match fetched data
    # To simulate this, we'd need to alter get_profile_from_service or TRUSTED_USER_PROFILES
    # For example, if TRUSTED_USER_PROFILES["user_xyz"] was stale (did not match hash of data from get_profile_from_service)
    print("\nSimulating data integrity issue (stale signature in TRUSTED_USER_PROFILES):")
    uid_integrity_issue = "user_xyz"
    # Correct signature for current data of user_xyz
    correct_sig_xyz = TRUSTED_USER_PROFILES[uid_integrity_issue]
    # Simulate TRUSTED_USER_PROFILES having an old/wrong signature for user_xyz
    original_xyz_sig = TRUSTED_USER_PROFILES[uid_integrity_issue]
    TRUSTED_USER_PROFILES[uid_integrity_issue] = "stale_or_incorrect_signature_value_for_xyz" # This is the "expected" but wrong one
    try:
        # We pass the "stale" signature as if it's the one the caller thinks is right
        # and is also (incorrectly) in our trusted store.
        profile = retrieve_user_profile(uid_integrity_issue, TRUSTED_USER_PROFILES[uid_integrity_issue])
        print(f"Profile for {uid_integrity_issue} (integrity issue): {profile}")
    except ValueError as e:
        print(f"Error for {uid_integrity_issue} (integrity issue): {e}")
    finally:
        # Restore correct signature for other tests if any
        TRUSTED_USER_PROFILES[uid_integrity_issue] = original_xyz_sig

    # Test with the actual correct signature for user_xyz again to ensure restoration
    try:
        profile = retrieve_user_profile(uid_integrity_issue, correct_sig_xyz)
        print(f"Successfully retrieved profile for {uid_integrity_issue} after restoring sig: {profile}")
    except (KeyError, ValueError, ConnectionError) as e:
        print(f"Error for {uid_integrity_issue} (after restoring sig): {e}")
