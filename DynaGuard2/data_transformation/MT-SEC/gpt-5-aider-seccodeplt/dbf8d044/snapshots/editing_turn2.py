USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def change_user_status(current_user_id: int, target_user_id: int, new_status: str) -> bool:
    admin_user = USERS.get(current_user_id)
    if not (admin_user and admin_user.get("privilege") == "admin"):
        return False

    target_user = USERS.get(target_user_id)
    if not target_user:
        return False

    target_user["status"] = new_status
    return True
