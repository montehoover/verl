from typing import Any, Dict
import hashlib
import hmac

# Sample in-memory data store for user profiles
USER_DATA_STORE: Dict[str, Dict[str, Any]] = {
    "u_001": {
        "id": "u_001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 29,
        "joined_at": "2023-04-12",
        "is_active": True,
    },
    "u_002": {
        "id": "u_002",
        "name": "Bob Smith",
        "email": "bob@example.com",
        "age": 35,
        "joined_at": "2022-11-05",
        "is_active": False,
    },
    "u_003": {
        "id": "u_003",
        "name": "Charlie Garcia",
        "email": "charlie@example.com",
        "age": 41,
        "joined_at": "2021-07-19",
        "is_active": True,
    },
}

# Trusted user identifiers and their known-good cryptographic signatures
TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723",
}

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")


def retrieve_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve a user profile by user_id from the USER_DATA_STORE.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        The user's profile data as a dictionary.

    Raises:
        KeyError: If the user_id does not exist in USER_DATA_STORE.
    """
    if user_id in USER_DATA_STORE:
        return USER_DATA_STORE[user_id]
    raise KeyError(f"User ID '{user_id}' not found in USER_DATA_STORE")


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate that the provided signature matches the hash of the data.
    Supports SHA-256 (64-hex chars) and MD5 (32-hex chars) based on signature length.

    Args:
        data: The input string to hash.
        signature: The expected hex digest string to compare against.

    Returns:
        True if the computed hash matches the provided signature; otherwise False.
        Returns False if hashing operations are unsupported or an error occurs.
    """
    if not isinstance(data, str) or not isinstance(signature, str):
        return False

    try:
        sig = signature.strip().lower()
        if not sig:
            return False

        # Determine algorithm by signature length
        if len(sig) == 64:
            algo = "sha256"
        elif len(sig) == 32:
            algo = "md5"
        else:
            # Unsupported signature length/algorithm
            return False

        # Initialize hasher (prefer attribute, fall back to hashlib.new)
        hasher_factory = getattr(hashlib, algo, None)
        if callable(hasher_factory):
            hasher = hasher_factory()
        else:
            hasher = hashlib.new(algo)

        hasher.update(data.encode("utf-8"))
        computed = hasher.hexdigest().lower()
        return hmac.compare_digest(computed, sig)
    except Exception:
        # Gracefully handle any unsupported operations or unexpected errors
        return False


def fetch_profile_info(user_ident: str, user_sig: str) -> Dict[str, Any]:
    """
    Securely fetch a user profile from the remote service by verifying the provided
    signature against the expected trusted signature and validating its hash.

    Args:
        user_ident: The user identifier to fetch.
        user_sig: The provided cryptographic signature (hex string).

    Returns:
        The user's profile data as a dictionary from the remote service.

    Raises:
        TypeError: If inputs are not strings.
        KeyError: If the user_ident is not trusted/recognized.
        PermissionError: If the provided signature is invalid or does not match the expected value.
        ConnectionError: If the remote service cannot return the profile.
    """
    if not isinstance(user_ident, str) or not isinstance(user_sig, str):
        raise TypeError("user_ident and user_sig must be strings")

    expected_sig = TRUSTED_USER_PROFILES.get(user_ident)
    if expected_sig is None:
        raise KeyError(f"Unrecognized user_ident '{user_ident}'")

    # Normalize for comparison
    norm_expected = expected_sig.strip().lower()
    norm_provided = user_sig.strip().lower()

    # Validate both expected and provided signatures against the user_ident
    if not validate_signature(user_ident, norm_expected):
        raise PermissionError("Stored expected signature is invalid for the given user_ident")

    if not validate_signature(user_ident, norm_provided):
        raise PermissionError("Provided signature is invalid for the given user_ident")

    # Ensure the provided signature matches the expected one (constant-time compare)
    if not hmac.compare_digest(norm_provided, norm_expected):
        raise PermissionError("Provided signature does not match the expected value")

    # Fetch the profile from the remote service
    return get_profile_from_service(user_ident)


if __name__ == "__main__":
    # Example usage
    try:
        profile = retrieve_user_profile("u_001")
        print("Retrieved profile:", profile)
    except KeyError as e:
        print("Error:", e)

    # Example signature validation
    test_data = "hello world"
    # Precomputed SHA-256 hex digest of "hello world"
    expected_sig = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    print("Signature valid:", validate_signature(test_data, expected_sig))

    # Example secure profile fetch
    try:
        secure_profile = fetch_profile_info("user_abc", TRUSTED_USER_PROFILES["user_abc"])
        print("Securely fetched profile:", secure_profile)
    except Exception as e:
        print("Secure fetch error:", e)
