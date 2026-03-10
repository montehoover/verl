import hashlib
from typing import Optional

USER_DATA_STORE = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com", "age": 30},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com", "age": 45},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com", "age": 8},
}

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_ident: str) -> Optional[dict]:
    if user_ident in TRUSTED_USER_PROFILES:
        # In a real scenario, this would fetch detailed profile data.
        # For this example, we return a generic profile.
        return {"name": "Test User", "email": f"{user_ident}@example.com", "id": user_ident}
    else:
        # This case should ideally not be hit if fetch_profile_info calls this,
        # as user_ident should already be validated against TRUSTED_USER_PROFILES.
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}: User not in trusted list (unexpected).")

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

def fetch_profile_info(user_ident: str, user_sig: str) -> Optional[dict]:
    """
    Securely fetches user profile information.

    It verifies the user_ident and user_sig against TRUSTED_USER_PROFILES,
    then validates the signature's cryptographic integrity using validate_signature,
    and finally fetches the profile from the service.

    Args:
        user_ident: The user identifier.
        user_sig: The provided signature for the user identifier.

    Returns:
        The user's profile data if all checks pass and fetching is successful,
        None otherwise.
    """
    expected_trusted_sig = TRUSTED_USER_PROFILES.get(user_ident)

    if expected_trusted_sig is None:
        print(f"Error (fetch_profile_info): User identifier '{user_ident}' not found in trusted list.")
        return None

    if user_sig != expected_trusted_sig:
        print(f"Error (fetch_profile_info): Provided signature for '{user_ident}' ('{user_sig}') does not match trusted signature ('{expected_trusted_sig}').")
        return None

    # At this point, user_sig matches the entry in TRUSTED_USER_PROFILES.
    # Now, integrate validate_signature to ensure this trusted signature
    # is a cryptographically valid SHA256 hash of the user_ident.
    if not validate_signature(user_ident, user_sig):
        print(f"CRITICAL Error (fetch_profile_info): The trusted signature for '{user_ident}' ('{user_sig}') is NOT a valid SHA256 hash "
              f"of the user identifier itself. TRUSTED_USER_PROFILES might be misconfigured or use a different hashing scheme than SHA256.")
        return None

    # All checks passed, proceed to fetch the profile from the service.
    try:
        profile_data = get_profile_from_service(user_ident)
        print(f"Successfully fetched profile for '{user_ident}'.")
        return profile_data
    except ConnectionError as e:
        print(f"Error (fetch_profile_info): Could not fetch profile for '{user_ident}' from service after successful validation: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"Unexpected error (fetch_profile_info) while fetching profile for '{user_ident}': {e}")
        return None

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

    print("\n--- Example usage for fetch_profile_info ---")

    # Case 1: User "user_abc" with their signature from TRUSTED_USER_PROFILES
    # This will fail the validate_signature check because the stored signature is not a SHA256 hash of "user_abc".
    print("\nFetching profile for 'user_abc' with its listed signature:")
    profile_abc = fetch_profile_info("user_abc", TRUSTED_USER_PROFILES["user_abc"])
    if profile_abc:
        print(f"Profile for user_abc: {profile_abc}")
    else:
        print("Failed to retrieve profile for user_abc.")

    # Case 2: User "user_xyz" with an incorrect signature
    print("\nFetching profile for 'user_xyz' with an incorrect signature:")
    profile_xyz_incorrect_sig = fetch_profile_info("user_xyz", "thisisdefinitelywrong")
    if profile_xyz_incorrect_sig:
        print(f"Profile for user_xyz (incorrect sig): {profile_xyz_incorrect_sig}")
    else:
        print("Failed to retrieve profile for user_xyz with incorrect signature.")

    # Case 3: Unknown user
    print("\nFetching profile for 'unknown_user':")
    profile_unknown = fetch_profile_info("unknown_user", "anyrandomsignature")
    if profile_unknown:
        print(f"Profile for unknown_user: {profile_unknown}")
    else:
        print("Failed to retrieve profile for unknown_user.")

    # Case 4: To demonstrate a fully successful path, we would need an entry in
    # TRUSTED_USER_PROFILES where the signature IS the SHA256 hash of the user_id.
    # Let's temporarily add one for demonstration if it's not too intrusive,
    # or just acknowledge this limitation of the current TRUSTED_USER_PROFILES data.
    print("\nDemonstrating a successful fetch with a correctly configured SHA256 signature:")
    temp_user_id = "temp_valid_user"
    temp_user_sig = hashlib.sha256(temp_user_id.encode('utf-8')).hexdigest()
    
    # Temporarily add to TRUSTED_USER_PROFILES for this test
    original_trusted_profiles = TRUSTED_USER_PROFILES.copy() # Keep a copy to restore
    TRUSTED_USER_PROFILES[temp_user_id] = temp_user_sig
    
    profile_temp_valid = fetch_profile_info(temp_user_id, temp_user_sig)
    if profile_temp_valid:
        print(f"Profile for {temp_user_id}: {profile_temp_valid}")
    else:
        print(f"Failed to retrieve profile for {temp_user_id} (should have succeeded).")
    
    # Restore original TRUSTED_USER_PROFILES
    TRUSTED_USER_PROFILES = original_trusted_profiles
    # Clean up the temporary user from the restored dictionary if it was shallow copied and modified
    if temp_user_id in TRUSTED_USER_PROFILES and temp_user_id not in original_trusted_profiles :
        del TRUSTED_USER_PROFILES[temp_user_id]


    # Example of get_profile_from_service directly (though usually not called this way)
    # print("\nDirect call to get_profile_from_service (for user in TRUSTED_USER_PROFILES):")
    # try:
    #     print(get_profile_from_service("user_abc"))
    # except ConnectionError as e:
    #     print(e)
    #
    # print("\nDirect call to get_profile_from_service (for user NOT in TRUSTED_USER_PROFILES):")
    # try:
    #     print(get_profile_from_service("user_not_trusted_directly"))
    # except ConnectionError as e:
    #     print(e)
