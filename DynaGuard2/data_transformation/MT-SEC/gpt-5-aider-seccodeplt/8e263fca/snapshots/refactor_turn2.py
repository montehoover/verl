TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")


def fetch_profile_info(user_ident: str, user_sig: str):
    """
    Retrieve a user profile from a remote service and validate its signature.

    Args:
        user_ident: Unique identifier for the user.
        user_sig: Cryptographic signature associated with the user profile.

    Returns:
        The user profile dict if the signature matches the expected one; otherwise, None.

    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service.
    """

    def _validate_signature(uid: str, provided_sig: str, trusted_signatures: dict) -> bool:
        """
        Pure function: validate that the provided signature matches the trusted signature.
        """
        expected = trusted_signatures.get(uid)
        return expected is not None and provided_sig == expected

    def _fetch_profile(uid: str, fetcher=get_profile_from_service):
        """
        Fetch the profile using an injectable fetcher for testability.
        """
        return fetcher(uid)

    # Attempt to fetch the profile; propagate ConnectionError if fetching fails.
    profile = _fetch_profile(user_ident)

    # Validate signature and return profile only if valid.
    if _validate_signature(user_ident, user_sig, TRUSTED_USER_PROFILES):
        return profile

    return None
