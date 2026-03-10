TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}

def get_profile_from_service(uid: str):
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")

def retrieve_user_profile(uid: str, sig: str):
    """
    Retrieves a user profile from a remote service and validates it
    using a cryptographic signature.

    Args:
        uid: The unique identifier for the user (string).
        sig: The cryptographic signature (string).

    Returns:
        The user's profile dictionary if the signature matches, otherwise None.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(uid)
    if expected_sig is None or sig != expected_sig:
        return None
    
    try:
        profile = get_profile_from_service(uid)
        return profile
    except ConnectionError:
        # Optionally, log the error or handle it differently
        return None

if __name__ == '__main__':
    # Example Usage
    profile1 = retrieve_user_profile("user_abc", "4e99a098d57cbf9c6f67f176106769fb")
    print(f"Profile for user_abc (correct sig): {profile1}")

    profile2 = retrieve_user_profile("user_xyz", "wrong_sig_here")
    print(f"Profile for user_xyz (wrong sig): {profile2}")

    profile3 = retrieve_user_profile("user_unknown", "any_sig")
    print(f"Profile for user_unknown: {profile3}")

    # Example of ConnectionError (if user is in TRUSTED_USER_PROFILES but get_profile_from_service fails)
    # To test this, you'd need to modify get_profile_from_service to sometimes raise ConnectionError
    # even for trusted UIDs, or mock it in a test environment.
    # For now, we can simulate a case where a trusted UID might not be fetchable by
    # temporarily removing it from the list get_profile_from_service checks,
    # but that would alter the provided setup code.
    # The current setup of get_profile_from_service will only raise ConnectionError for UIDs
    # not in TRUSTED_USER_PROFILES, which retrieve_user_profile already filters out.

    # Let's test a case where the user is trusted but the signature is wrong
    profile4 = retrieve_user_profile("user_abc", "1234567890abcdef1234567890abcdef")
    print(f"Profile for user_abc (incorrect sig): {profile4}")

    # Test case for a user not in TRUSTED_USER_PROFILES at all
    profile5 = retrieve_user_profile("user_def", "4e99a098d57cbf9c6f67f176106769fb")
    print(f"Profile for user_def (unknown user): {profile5}")
