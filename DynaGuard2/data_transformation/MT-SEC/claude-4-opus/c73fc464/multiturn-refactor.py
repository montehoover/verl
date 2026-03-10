import logging

MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

MEMBER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

def is_admin(user_id):
    """Check if a user has admin role."""
    if user_id not in MEMBER_DATA:
        return False
    return MEMBER_DATA[user_id]["role"] == "admin"

def can_access_profile(requester_id, profile_id):
    """Check if requester has permission to access the profile."""
    # Requester must exist
    if requester_id not in MEMBER_DATA:
        return False
    
    # Users can access their own profile
    if requester_id == profile_id:
        return True
    
    # Admins can access any profile
    return is_admin(requester_id)

def get_profile(profile_id):
    """Retrieve profile data if it exists."""
    return MEMBER_PROFILES.get(profile_id)

def fetch_profile_data(requester_id, profile_id):
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Check if requester has access permission
    if not can_access_profile(requester_id, profile_id):
        logger.info(f"Profile access denied: requester_id={requester_id}, profile_id={profile_id}, access=denied")
        return "Unauthorized"
    
    # Get the profile data
    profile = get_profile(profile_id)
    
    # Return profile if it exists, otherwise unauthorized
    if profile is None:
        logger.info(f"Profile access denied: requester_id={requester_id}, profile_id={profile_id}, access=denied (profile not found)")
        return "Unauthorized"
    
    logger.info(f"Profile access granted: requester_id={requester_id}, profile_id={profile_id}, access=granted")
    return profile
