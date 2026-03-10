from typing import Dict, Any


def create_user_profile(uid: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile by combining a unique identifier with user details.
    
    Args:
        uid: A unique identifier for the user
        details: A dictionary containing user attributes
        
    Returns:
        A dictionary containing the complete user profile
    """
    user_profile = {
        'uid': uid,
        **details
    }
    return user_profile
