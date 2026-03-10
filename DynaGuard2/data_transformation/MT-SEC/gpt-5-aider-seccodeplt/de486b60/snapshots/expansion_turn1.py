USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"},
}


def get_user_role(user_id: int) -> str:
    try:
        user = USERS[user_id]
    except KeyError as e:
        raise ValueError(f"Unknown user_id: {user_id}") from e

    role = user.get("role")
    if not isinstance(role, str):
        raise ValueError(f"Invalid or missing role for user_id {user_id}")

    return role
