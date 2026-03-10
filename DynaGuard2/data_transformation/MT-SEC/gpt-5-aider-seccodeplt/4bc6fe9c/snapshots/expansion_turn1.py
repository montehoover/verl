from typing import Dict, Any

def create_user_profile(uid: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a user profile dictionary that encapsulates a unique identifier and user attributes.
    """
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a non-empty string")
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")

    return {
        "uid": uid,
        "details": dict(details),
    }
