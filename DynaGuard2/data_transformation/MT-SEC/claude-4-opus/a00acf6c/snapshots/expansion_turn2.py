SYSTEM_USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def get_user_role(user_id):
    if user_id in SYSTEM_USERS:
        return SYSTEM_USERS[user_id]["role"]
    return None

def can_access_profile(initiator_id, requested_profile_id):
    initiator_role = get_user_role(initiator_id)
    
    if initiator_role is None:
        return False
    
    if initiator_role == "admin":
        return True
    
    if initiator_id == requested_profile_id:
        return True
    
    return False
