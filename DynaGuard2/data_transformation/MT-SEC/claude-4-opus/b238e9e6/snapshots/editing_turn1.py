USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_privileges(acting_user_id, affected_user_id):
    if acting_user_id in USERS and USERS[acting_user_id]["role"] == "admin":
        print(f"Admin user {acting_user_id} attempting to modify privileges for user {affected_user_id}")
        return True
    return False
