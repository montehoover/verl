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


def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    allowed_privileges = {"normal", "moderator", "admin"}

    actor = USERS.get(actor_id)
    if not (actor and actor.get("privilege") == "admin"):
        return False

    recipient = USERS.get(recipient_id)
    if not recipient:
        return False

    if updated_privilege not in allowed_privileges:
        return False

    recipient["privilege"] = updated_privilege
    return True
