from typing import Any, Dict, Iterable, Optional

# Global data structure tracking user roles: maps user_id (int) to role(s).
# It can be updated elsewhere in the application.
# Accepted shapes per user_id:
# - str: a single role name
# - Iterable[str]: a collection of role names (e.g., list, set, tuple)
# - dict: a mapping of role name -> truthy/falsy flag
USER_ROLES: Dict[int, Any] = {}

# Global data structure tracking user subscription levels.
# Maps user_id (int) to subscription info, which can be:
# - str: the level name ('basic' or 'premium')
# - dict: may contain 'level' -> str, or keys 'basic'/'premium' -> truthy/falsy
# - Iterable[str]: a collection of level names; highest recognized level is used
SUBSCRIPTIONS: Dict[int, Any] = {}

_LEVEL_ORDER = {"basic": 0, "premium": 1}
_TRUTHY_STRINGS = {"1", "true", "yes", "y", "on"}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    return bool(value)


def _level_rank(level: str) -> Optional[int]:
    if not isinstance(level, str):
        return None
    key = level.strip().lower()
    return _LEVEL_ORDER.get(key)


def is_admin(user_id: int) -> bool:
    """
    Return True if the given user_id has an admin role, otherwise False.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    roles = USER_ROLES.get(user_id)
    if roles is None:
        return False

    # Single role as string
    if isinstance(roles, str):
        return roles.lower() == "admin"

    # Multiple roles as list/set/tuple of strings
    if isinstance(roles, Iterable) and not isinstance(roles, (str, bytes, dict)):
        for r in roles:
            if isinstance(r, str) and r.lower() == "admin":
                return True
        return False

    # Roles as a mapping: role_name -> flag
    if isinstance(roles, dict):
        val = roles.get("admin")
        if val is None:
            return False
        # Interpret typical truthy values
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(val)

    return False


def has_access_level(user_id: int, required_level: str) -> bool:
    """
    Return True if the user's subscription level meets or exceeds the required level.

    Arguments:
    - user_id: int
    - required_level: str, either 'basic' or 'premium' (case-insensitive)
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")
    if not isinstance(required_level, str):
        raise TypeError("required_level must be a str")

    req_rank = _level_rank(required_level)
    if req_rank is None:
        raise ValueError("required_level must be 'basic' or 'premium'")

    sub = SUBSCRIPTIONS.get(user_id)
    if sub is None:
        return False

    user_rank: Optional[int] = None

    # Single level as string
    if isinstance(sub, str):
        user_rank = _level_rank(sub)

    # Mapping forms
    elif isinstance(sub, dict):
        # Prefer explicit 'level'
        if "level" in sub:
            user_rank = _level_rank(sub.get("level"))  # type: ignore[arg-type]
        # Otherwise infer from truthy flags
        for level_name in _LEVEL_ORDER:
            if level_name in sub and _is_truthy(sub[level_name]):
                rank = _level_rank(level_name)
                if rank is not None and (user_rank is None or rank > user_rank):
                    user_rank = rank

    # Iterable of strings (e.g., ['basic'])
    elif isinstance(sub, Iterable) and not isinstance(sub, (str, bytes, dict)):
        for item in sub:
            if isinstance(item, str):
                rank = _level_rank(item)
                if rank is not None and (user_rank is None or rank > user_rank):
                    user_rank = rank

    if user_rank is None:
        return False

    return user_rank >= req_rank
