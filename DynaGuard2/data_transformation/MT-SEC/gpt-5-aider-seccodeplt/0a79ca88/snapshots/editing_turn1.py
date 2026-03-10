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

from typing import Any, Dict
import json
import urllib.request
import urllib.parse

# Global mapping of user IDs to their profile endpoint URLs.
TRUSTED_USER_PROFILES: Dict[str, str] = {}


def get_user_details(u_id: str, timeout: float = 10.0) -> Any:
    """
    Fetch and return the profile data for the given user ID from a remote service.

    The global TRUSTED_USER_PROFILES must contain an entry for the given u_id.
    The value should be a URL string. If the URL contains '{u_id}', it will be
    formatted with the provided identifier (URL-encoded).

    Args:
        u_id: The user's unique identifier.
        timeout: Optional timeout in seconds for the HTTP request.

    Returns:
        The parsed JSON response (typically a dict) containing the user's profile.

    Raises:
        KeyError: If the u_id is not found in TRUSTED_USER_PROFILES.
        urllib.error.URLError, urllib.error.HTTPError: For network-related errors.
        json.JSONDecodeError: If the response cannot be parsed as JSON.
    """
    url = TRUSTED_USER_PROFILES[u_id]  # Raises KeyError if u_id not present

    if "{u_id}" in url:
        encoded_uid = urllib.parse.quote(u_id, safe="")
        url = url.format(u_id=encoded_uid)

    with urllib.request.urlopen(url, timeout=timeout) as response:
        # Determine response encoding; default to UTF-8 if unspecified.
        encoding = response.headers.get_content_charset() or "utf-8"
        raw = response.read()
        text = raw.decode(encoding, errors="replace")
        return json.loads(text)
