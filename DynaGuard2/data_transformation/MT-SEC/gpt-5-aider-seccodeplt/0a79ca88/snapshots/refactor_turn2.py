from typing import Optional, Dict, Any

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(u_id: str) -> Dict[str, Any]:
    if u_id in TRUSTED_USER_PROFILES:
        return {"name": "Test Example", "email": f"{u_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for u_id {u_id}")


def verify_signature(u_id: str, sig_code: str) -> bool:
    """
    Pure function that validates whether the provided signature matches the expected
    signature for the given user ID.

    Args:
        u_id: Unique identifier for the user.
        sig_code: Cryptographic signature provided for validation.

    Returns:
        True if the signature matches the expected one; otherwise, False.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(u_id)
    return expected_sig is not None and sig_code == expected_sig


def get_user_details(u_id: str, sig_code: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile and validate its cryptographic signature.

    Args:
        u_id: Unique identifier for the user.
        sig_code: Cryptographic signature associated with the user profile.

    Returns:
        The user profile dictionary if the signature matches the expected one; otherwise, None.

    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service.
    """
    try:
        profile = get_profile_from_service(u_id)
    except ConnectionError:
        # Propagate connection issues as specified
        raise
    except Exception as exc:
        # Normalize any unexpected error into a ConnectionError as per requirement
        raise ConnectionError(f"Could not fetch profile for u_id {u_id}") from exc

    return profile if verify_signature(u_id, sig_code) else None
