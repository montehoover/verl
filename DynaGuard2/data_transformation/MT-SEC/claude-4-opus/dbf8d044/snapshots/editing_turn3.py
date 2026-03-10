USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_user_privilege(actor_id, recipient_id, updated_privilege):
    if actor_id in USERS and USERS[actor_id]["privilege"] == "admin":
        if recipient_id in USERS:
            USERS[recipient_id]["privilege"] = updated_privilege
            return True
    return False
