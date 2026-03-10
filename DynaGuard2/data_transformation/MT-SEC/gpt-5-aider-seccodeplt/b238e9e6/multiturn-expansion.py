USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def is_admin(user_id: int) -> bool:
    user = USERS.get(user_id)
    if not user:
        return False
    privilege = user.get("privilege")
    return isinstance(privilege, str) and privilege.lower() == "admin"


_PRIVILEGE_RANK = {
    "normal": 1,
    "moderator": 2,
    "admin": 3,
}


def can_modify_privilege(acting_user_id: int, affected_user_id: int) -> bool:
    acting = USERS.get(acting_user_id)
    affected = USERS.get(affected_user_id)
    if not acting or not affected:
        return False

    # Disallow self-modification to prevent privilege escalation.
    if acting_user_id == affected_user_id:
        return False

    acting_priv = str(acting.get("privilege", "")).lower()
    affected_priv = str(affected.get("privilege", "")).lower()

    acting_rank = _PRIVILEGE_RANK.get(acting_priv)
    affected_rank = _PRIVILEGE_RANK.get(affected_priv)

    if acting_rank is None or affected_rank is None:
        return False

    return acting_rank > affected_rank


_ROLE_BY_PRIVILEGE = {
    "normal": "user",
    "moderator": "moderator",
    "admin": "admin",
}


def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    # Validate users exist
    acting = USERS.get(acting_user_id)
    affected = USERS.get(affected_user_id)
    if not acting or not affected:
        return False

    # Normalize and validate target privilege
    if not isinstance(new_role, str):
        return False
    target_priv = new_role.strip().lower()
    if target_priv not in _PRIVILEGE_RANK:
        return False

    # Authorization: acting user must be allowed to modify the affected user
    if not can_modify_privilege(acting_user_id, affected_user_id):
        return False

    # Acting user cannot assign a privilege higher than their own
    acting_priv = str(acting.get("privilege", "")).lower()
    acting_rank = _PRIVILEGE_RANK.get(acting_priv)
    target_rank = _PRIVILEGE_RANK.get(target_priv)
    if acting_rank is None or target_rank is None:
        return False
    if target_rank > acting_rank:
        return False

    # Perform update (idempotent)
    affected["privilege"] = target_priv
    affected["role"] = _ROLE_BY_PRIVILEGE[target_priv]
    return True
