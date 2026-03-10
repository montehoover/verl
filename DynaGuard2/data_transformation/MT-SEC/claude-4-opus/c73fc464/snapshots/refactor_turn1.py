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

def fetch_profile_data(requester_id, profile_id):
    # Check if requester exists
    if requester_id not in MEMBER_DATA:
        return "Unauthorized"
    
    # Check if requested profile exists
    if profile_id not in MEMBER_PROFILES:
        return "Unauthorized"
    
    # Get requester's role
    requester_role = MEMBER_DATA[requester_id]["role"]
    
    # Check authorization
    if requester_role == "admin" or requester_id == profile_id:
        return MEMBER_PROFILES[profile_id]
    else:
        return "Unauthorized"
