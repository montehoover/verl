TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(user_identifier: str) -> dict | None:
    """
    Simulates fetching a user profile from an external service.
    """
    if user_identifier in TRUSTED_USER_PROFILES:
        # In a real scenario, this data would come from an external call
        return {"name": f"User {user_identifier.split('_')[-1].upper()}", "email": f"{user_identifier}@example.com", "id": user_identifier}
    else:
        # Simulate a failure to fetch for untrusted or non-existent users in the service
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier} from service.")

def collect_user_data(user_identifier: str, user_signature: str) -> dict | None:
    """
    Retrieves a user profile from an external service and verifies it
    using a cryptographic signature.

    Args:
        user_identifier: The ID of the user whose profile is being fetched.
        user_signature: The cryptographic signature for the user's profile data.

    Returns:
        The profile data as a dictionary if the signature is verified
        and the profile is successfully fetched, otherwise None.
    """
    expected_signature = TRUSTED_USER_PROFILES.get(user_identifier)

    if expected_signature is None:
        print(f"Debug: User ID '{user_identifier}' not found in TRUSTED_USER_PROFILES.")
        return None

    if user_signature != expected_signature:
        print(f"Debug: Signature mismatch for user ID '{user_identifier}'. Expected '{expected_signature}', got '{user_signature}'.")
        return None

    try:
        profile_data = get_profile_from_service(user_identifier)
        # In a real application, you might perform further validation on profile_data here
        # or ensure the signature actually corresponds to the *content* of profile_data,
        # not just the user_identifier. For this example, we assume the signature
        # in TRUSTED_USER_PROFILES is for the act of fetching this user's data.
        return profile_data
    except ConnectionError as e:
        print(f"Error: Could not retrieve profile for '{user_identifier}': {e}")
        return None

if __name__ == '__main__':
    # Example Usage

    # Case 1: Valid user, correct signature
    print("Case 1: Valid user, correct signature (user_abc)")
    profile_abc = collect_user_data("user_abc", "4e99a098d57cbf9c6f67f176106769fb")
    if profile_abc:
        print(f"  Profile: {profile_abc}\n")
    else:
        print("  Failed to retrieve profile.\n")

    # Case 2: Valid user, incorrect signature
    print("Case 2: Valid user, incorrect signature (user_xyz)")
    profile_xyz_bad_sig = collect_user_data("user_xyz", "incorrect_signature_here")
    if profile_xyz_bad_sig:
        print(f"  Profile: {profile_xyz_bad_sig}\n")
    else:
        print("  Failed to retrieve profile.\n")

    # Case 3: Invalid user (not in TRUSTED_USER_PROFILES)
    print("Case 3: Invalid user (user_def)")
    profile_def = collect_user_data("user_def", "any_signature")
    if profile_def:
        print(f"  Profile: {profile_def}\n")
    else:
        print("  Failed to retrieve profile.\n")

    # Case 4: User in TRUSTED_USER_PROFILES but get_profile_from_service might fail
    # To test this, we'd need to modify get_profile_from_service or TRUSTED_USER_PROFILES
    # For example, if "user_fail" was in TRUSTED_USER_PROFILES but not handled by get_profile_from_service
    # (current setup ensures this won't happen unless ConnectionError is raised for other reasons)
    # Let's add a temporary user to TRUSTED_USER_PROFILES that get_profile_from_service will reject
    
    print("Case 4: Simulating service connection error for a trusted user")
    # Temporarily add a user that will cause get_profile_from_service to raise an error
    # by not being in its "known" list for successful profile fetching,
    # even if it's in TRUSTED_USER_PROFILES.
    # To do this properly, we'd need to make get_profile_from_service behave differently.
    # The current get_profile_from_service will succeed if user_identifier is in TRUSTED_USER_PROFILES.
    # Let's simulate by trying to fetch a user not in TRUSTED_USER_PROFILES for the ConnectionError path
    # This is slightly different from the description, but shows the ConnectionError handling.
    
    # To truly test the ConnectionError for a *trusted* user, we'd modify get_profile_from_service
    # to sometimes fail even for trusted users.
    # For now, let's demonstrate the ConnectionError path by calling get_profile_from_service directly
    # with an ID that will cause it to raise an error.
    # The collect_user_data function itself won't hit this for "user_service_fail"
    # because "user_service_fail" is not in TRUSTED_USER_PROFILES.

    # Let's adjust the test to be more direct for collect_user_data:
    # Add a user to TRUSTED_USER_PROFILES that get_profile_from_service will fail for.
    # This requires get_profile_from_service to be able to fail for users it "knows" about.
    # The current setup of get_profile_from_service doesn't allow this easily.
    # It only raises ConnectionError if user_identifier is NOT in TRUSTED_USER_PROFILES.

    # A better test for ConnectionError within collect_user_data:
    # Assume 'user_service_error' is trusted, but the service call fails.
    TRUSTED_USER_PROFILES["user_service_error"] = "signature_for_service_error"
    
    # We need to modify get_profile_from_service to simulate this specific failure
    # For simplicity, let's assume any call to get_profile_from_service with 'user_service_error'
    # will now raise a ConnectionError, even if it's "trusted".
    # This is a bit of a hack for testing without changing the function signature.
    
    # A more realistic get_profile_from_service for this test:
    _original_get_profile = get_profile_from_service
    def mock_get_profile_from_service_for_error(user_identifier: str):
        if user_identifier == "user_service_error":
            raise ConnectionError(f"Simulated service outage for user_id {user_identifier}")
        return _original_get_profile(user_identifier)

    _real_get_profile_func = get_profile_from_service # Save real one
    get_profile_from_service = mock_get_profile_from_service_for_error # Monkeypatch

    print("Case 4: Trusted user, but service connection error (user_service_error)")
    profile_service_error = collect_user_data("user_service_error", "signature_for_service_error")
    if profile_service_error:
        print(f"  Profile: {profile_service_error}\n")
    else:
        print("  Failed to retrieve profile.\n")
    
    get_profile_from_service = _real_get_profile_func # Restore real one
    del TRUSTED_USER_PROFILES["user_service_error"] # Clean up
