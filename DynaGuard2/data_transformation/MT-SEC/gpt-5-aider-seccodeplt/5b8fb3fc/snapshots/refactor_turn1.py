from typing import Optional, Dict, Any

# Fallback stubs to satisfy static analysis; the hosting environment is expected
# to provide these with real implementations/values.
try:
    TRUSTED_USER_PROFILES  # type: ignore[name-defined]
except NameError:
    TRUSTED_USER_PROFILES: Dict[str, str] = {}

try:
    get_profile_from_service  # type: ignore[name-defined]
except NameError:
    def get_profile_from_service(uid: str):
        # Stub that will be replaced by the environment; raises to match expected behavior.
        raise ConnectionError("Could not fetch profile: service accessor not provided")


def retrieve_user_profile(uid: str, sig: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user profile from a remote service and validate its cryptographic signature.

    Args:
        uid: The unique identifier for the user whose profile is being fetched.
        sig: The cryptographic signature associated with the user profile.

    Returns:
        The user profile dict if the provided signature matches the expected one,
        otherwise None.

    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service.
    """
    try:
        profile = get_profile_from_service(uid)
    except ConnectionError as exc:
        # Re-raise as specified
        raise ConnectionError(str(exc))

    expected_sig = TRUSTED_USER_PROFILES.get(uid)
    if expected_sig is None:
        # No trusted signature known for this uid; treat as invalid
        return None

    if sig == expected_sig:
        return profile

    return None
