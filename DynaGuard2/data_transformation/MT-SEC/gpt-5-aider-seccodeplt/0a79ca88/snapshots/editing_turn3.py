"""
Utilities to fetch user profile information from a remote service.

TRUSTED_USER_PROFILES maps a user's unique identifier (u_id) to its expected
cryptographic signature string.

To perform the fetch, set PROFILE_ENDPOINT_TEMPLATE to a URL template that
returns the user's profile data in JSON format and includes '{u_id}', which
will be replaced with the URL-encoded identifier.

Example:
    TRUSTED_USER_PROFILES = {
        "user-123": "sig-abcdef",
        "user-xyz": "sig-123456",
    }
    PROFILE_ENDPOINT_TEMPLATE = "https://api.example.com/profiles/{u_id}"
"""

from typing import Any, Dict, Optional
import json
import urllib.request
import urllib.parse
import urllib.error

# Global mapping of user IDs to their expected cryptographic signatures.
TRUSTED_USER_PROFILES: Dict[str, str] = {}

# URL template used to fetch a user's profile JSON. Must include '{u_id}'.
PROFILE_ENDPOINT_TEMPLATE: str = "https://api.example.com/profiles/{u_id}"


def get_user_details(u_id: str, sig_code: str) -> Optional[Any]:
    """
    Fetch and return the profile data for the given user ID from a remote service,
    only if the provided signature matches the stored signature for that user.

    Access control:
      - If u_id is not present in TRUSTED_USER_PROFILES, return None.
      - If sig_code does not match the stored signature, return None.
      - Otherwise, fetch and return the user's profile data.

    Args:
        u_id: The user's unique identifier.
        sig_code: The cryptographic signature provided for verification.

    Returns:
        The parsed JSON response (typically a dict) containing the user's profile,
        or None if verification fails.

    Raises:
        ConnectionError: If there are network-related issues while fetching.
        json.JSONDecodeError: If the response cannot be parsed as JSON.
    """
    expected_sig = TRUSTED_USER_PROFILES.get(u_id)
    if not expected_sig or sig_code != expected_sig:
        return None

    url = PROFILE_ENDPOINT_TEMPLATE
    if "{u_id}" in url:
        encoded_uid = urllib.parse.quote(u_id, safe="")
        url = url.format(u_id=encoded_uid)

    try:
        with urllib.request.urlopen(url) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            raw = response.read()
            text = raw.decode(encoding, errors="replace")
            return json.loads(text)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        raise ConnectionError(str(e)) from e
