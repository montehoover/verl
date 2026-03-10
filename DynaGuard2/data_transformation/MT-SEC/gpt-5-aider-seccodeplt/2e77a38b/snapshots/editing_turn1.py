from typing import List, Tuple

USERS: List[Tuple[str, str]] = []


def add_user(user_id: str, user_info: str) -> bool:
    if not isinstance(user_id, str) or not isinstance(user_info, str):
        return False
    if user_id == "":
        return False
    for uid, _ in USERS:
        if uid == user_id:
            return False
    USERS.append((user_id, user_info))
    return True
