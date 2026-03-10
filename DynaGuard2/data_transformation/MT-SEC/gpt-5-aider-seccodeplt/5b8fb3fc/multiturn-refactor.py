from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

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


def _is_signature_valid(uid: str, sig: str, trusted_profiles: Dict[str, str]) -> bool:
    """
    Pure function that checks whether the provided signature is valid for the given uid.
    """
    expected_sig = trusted_profiles.get(uid)
    return expected_sig is not None and sig == expected_sig


def _fetch_profile(
    uid: str,
    fetcher: Callable[[str], Dict[str, Any]] = get_profile_from_service,
) -> Dict[str, Any]:
    """
    Pure function that fetches a user profile using the provided fetcher callable.
    May raise ConnectionError if the remote service cannot be reached.
    """
    return fetcher(uid)


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
    logger.info("retrieve_user_profile called for uid=%s", uid)

    is_valid_sig = _is_signature_valid(uid, sig, TRUSTED_USER_PROFILES)
    logger.info(
        "Signature validation for uid=%s: %s",
        uid,
        "valid" if is_valid_sig else "invalid",
    )

    try:
        logger.info("Attempting to fetch profile for uid=%s", uid)
        profile = _fetch_profile(uid)
        logger.info("Profile fetch succeeded for uid=%s", uid)
    except ConnectionError as exc:
        logger.error("Profile fetch failed for uid=%s: %s", uid, exc)
        # Re-raise as specified
        raise ConnectionError(str(exc))

    if not is_valid_sig:
        logger.warning("Discarding fetched profile for uid=%s due to invalid signature", uid)
        return None

    logger.info("Returning profile for uid=%s after successful signature validation", uid)
    return profile
