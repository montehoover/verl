"""
Utilities to fetch user profile information from a remote service.

TRUSTED_USER_PROFILES should map a user's unique identifier (u_id) to a URL
endpoint that returns that user's profile data in JSON format. The URL can be
a fully resolved endpoint or a template containing '{u_id}' which will be
formatted with the provided identifier.

Example:
    TRUSTED_USER_PROFILES = {
        "user-123": "https://api.example.com/profiles/user-123",
        "user-xyz": "https://api.example.com/profiles/{u_id}",
    }
"""

from typing import Any, Dict, Optional
import json
import urllib.request
import urllib.parse
import urllib.error

# Global mapping of user IDs to their profile endpoint URLs.
TRUSTED_USER_PROFILES: Dict[str, str] = {}


def get_user_details(u_id: str, is_verified: bool) -> Optional[Any]:
    """
    Fetch and return the profile data for the given user ID from a remote service,
    subject to access control.

    Access control:
      - If is_verified is False, return None without contacting the remote service.
      - If is_verified is True but u_id is not in TRUSTED_USER_PROFILES, return None.
      - Otherwise, fetch and return the user's profile data.

    Args:
        u_id: The user's unique identifier.
        is_verified: Whether the caller has been verified for access.

    Returns:
        The parsed JSON response (typically a dict) containing the user's profile,
        or None if access is not permitted or if an error occurs while fetching.

    Notes:
        Network and parsing errors are handled and result in None being returned.
    """
    if not is_verified:
        return None

    url = TRUSTED_USER_PROFILES.get(u_id)
    if not url:
        return None

    if "{u_id}" in url:
        encoded_uid = urllib.parse.quote(u_id, safe="")
        url = url.format(u_id=encoded_uid)

    try:
        with urllib.request.urlopen(url) as response:
            # Determine response encoding; default to UTF-8 if unspecified.
            encoding = response.headers.get_content_charset() or "utf-8"
            raw = response.read()
            text = raw.decode(encoding, errors="replace")
            return json.loads(text)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None
