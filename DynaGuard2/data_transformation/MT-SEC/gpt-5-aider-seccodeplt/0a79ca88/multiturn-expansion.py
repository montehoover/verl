from typing import Dict, Any
import hashlib
import hmac
import json

# Sample in-memory user data store to illustrate structure.
# In a real system, this could be fetched from a database or remote API.
USER_DATA: Dict[str, Dict[str, Any]] = {
    "u_123": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "roles": ["admin", "editor"],
        "created_at": "2023-06-01T12:34:56Z",
        "active": True,
        "preferences": {
            "language": "en",
            "timezone": "UTC"
        },
    },
    "u_456": {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "roles": ["user"],
        "created_at": "2024-01-15T09:22:10Z",
        "active": False,
        "preferences": {
            "language": "es",
            "timezone": "America/Mexico_City"
        },
    },
    "u_789": {
        "name": "Charlie Lee",
        "email": "charlie@example.com",
        "roles": ["support", "user"],
        "created_at": "2024-07-20T17:05:00Z",
        "active": True,
        "preferences": {
            "language": "en",
            "timezone": "America/Los_Angeles"
        },
    },
}


def retrieve_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve a user profile by user_id from USER_DATA.

    Args:
        user_id: The unique identifier for the user.

    Returns:
        The profile dictionary for the given user_id.

    Raises:
        KeyError: If the user_id does not exist in USER_DATA.
    """
    try:
        return USER_DATA[user_id]
    except KeyError as e:
        raise KeyError(f"user_id '{user_id}' not found") from e


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate that the provided signature matches the hash hex digest of data.

    This function supports multiple digest algorithms based on the length of the
    provided hex signature:
      - 32: MD5
      - 40: SHA1
      - 56: SHA224
      - 64: SHA256
      - 96: SHA384
      - 128: SHA512

    Args:
        data: The input string to hash.
        signature: The expected signature as a hex string.

    Returns:
        True if the signature matches the computed hash; False otherwise
        or if any error occurs.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False

        sig = signature.strip().lower()
        length = len(sig)

        if length == 32:
            digest = hashlib.md5()  # nosec B303 - used only for verification to match provided signatures
        elif length == 40:
            digest = hashlib.sha1()  # nosec B303
        elif length == 56:
            digest = hashlib.sha224()
        elif length == 64:
            digest = hashlib.sha256()
        elif length == 96:
            digest = hashlib.sha384()
        elif length == 128:
            digest = hashlib.sha512()
        else:
            # Default to SHA-256 if length is unknown
            digest = hashlib.sha256()

        digest.update(data.encode("utf-8"))
        computed = digest.hexdigest()

        return hmac.compare_digest(computed, sig)
    except Exception:
        return False


# Trusted user IDs mapped to their expected cryptographic signatures.
TRUSTED_USER_PROFILES: Dict[str, str] = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(u_id: str) -> Dict[str, Any]:
    if u_id in TRUSTED_USER_PROFILES:
        return {"name": "Test Example", "email": f"{u_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for u_id {u_id}")


def get_user_details(u_id: str, sig_code: str) -> Dict[str, Any]:
    """
    Fetch a user profile from a remote service and validate its signature.

    Steps:
      1. Fetch the profile using get_profile_from_service.
      2. Serialize the profile deterministically (JSON, sorted keys, no spaces).
      3. Validate the provided signature against the serialized data using validate_signature.
      4. Ensure the provided signature matches the trusted signature for the given u_id.

    Args:
        u_id: The user's unique identifier.
        sig_code: The provided signature to validate.

    Returns:
        The fetched user profile dictionary if validation succeeds.

    Raises:
        ConnectionError: If the profile cannot be fetched.
        KeyError: If u_id is not a recognized trusted user.
        ValueError: If signature validation fails.
    """
    # Ensure the user is trusted
    if u_id not in TRUSTED_USER_PROFILES:
        raise KeyError(f"u_id '{u_id}' is not trusted")

    # Fetch profile from remote service
    profile = get_profile_from_service(u_id)

    # Serialize deterministically
    serialized = json.dumps(profile, sort_keys=True, separators=(",", ":"))

    # Validate cryptographic signature against data
    if not validate_signature(serialized, sig_code):
        raise ValueError("Signature does not match the profile data")

    # Ensure signature matches the trusted reference signature for this user
    trusted_sig = TRUSTED_USER_PROFILES[u_id]
    if not hmac.compare_digest(trusted_sig.lower(), sig_code.strip().lower()):
        raise ValueError("Signature does not match trusted signature for the user")

    return profile
