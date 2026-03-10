from typing import Dict, Optional, Set, Any, Iterable, List
from datetime import date
from urllib.parse import urlparse

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # Fallback when not available


# In-memory user store for demonstration.
# Replace this with your persistent data source (e.g., database, API).
_USER_STORE: Dict[str, Dict[str, object]] = {
    # user_id: {"active": bool, "roles": set[str]}
    "alice": {"active": True, "roles": {"owner", "admin"}},
    "bob": {"active": True, "roles": {"manager"}},
    "carol": {"active": True, "roles": {"member"}},
    "dave": {"active": False, "roles": {"viewer"}},
}

# In-memory projects database for demonstration.
PROJECTS_DB: Dict[str, Dict[str, Any]] = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"},
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"},
    },
}


def _normalize_str(value: str) -> str:
    """Normalize string inputs for consistent comparison."""
    return value.strip().lower()


def _get_user_record(user_id: str) -> Optional[Dict[str, object]]:
    """Fetch a user's record from the user store."""
    return _USER_STORE.get(user_id)


def authenticate_user(user_id: str, role: str) -> bool:
    """
    Determine if a user has access rights for a given role.

    Rules:
    - user_id and role must be non-empty strings
    - user must exist in the system
    - user must be active
    - user must possess the requested role

    Args:
        user_id: The unique identifier of the user.
        role: The role to check for access.

    Returns:
        True if the user is a valid, active member with the specified role; otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(role, str):
        return False

    user_id_norm = _normalize_str(user_id)
    role_norm = _normalize_str(role)

    if not user_id_norm or not role_norm:
        return False

    user = _get_user_record(user_id_norm)
    if not user:
        return False

    if not user.get("active", False):
        return False

    roles: Set[str] = {r.lower() for r in (user.get("roles") or set())}
    return role_norm in roles


# ----- Project settings validation helpers and rules -----

_KNOWN_VISIBILITIES: Set[str] = {"private", "internal", "public"}
_KNOWN_ROLES: Set[str] = {"owner", "admin", "manager", "member", "viewer"}
_ALLOWED_KEYS: Set[str] = {
    "name",
    "visibility",
    "max_members",
    "archived",
    "members",
    "owners",
    "start_date",
    "due_date",
    "timezone",
    "version",
    "tags",
    "webhooks",
    "sprints_enabled",
    "sprint_length_days",
    "allowed_roles",
}
_IMMUTABLE_KEYS: Set[str] = {"project_id", "created_at", "created_by"}


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _parse_iso_date(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        # Accept YYYY-MM-DD format
        return date.fromisoformat(value)
    except Exception:
        return None


def _is_valid_timezone(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if ZoneInfo is None:
        # Fallback heuristic when zoneinfo is unavailable
        return "/" in value
    try:
        ZoneInfo(value)
        return True
    except Exception:
        return False


def _collect_user_ids(raw: Any) -> Optional[Set[str]]:
    """
    Convert a raw list/set/tuple of user identifiers into a normalized set
    after validating that each user exists and is active.
    """
    if isinstance(raw, (list, set, tuple)):
        normalized: Set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                return None
            uid = _normalize_str(item)
            if not uid:
                return None
            record = _get_user_record(uid)
            if not record or not record.get("active", False):
                return None
            normalized.add(uid)
        return normalized
    return None


def validate_project_settings(current_settings: Dict[str, Any], settings_update: Dict[str, Any]) -> bool:
    """
    Validate a proposed update to project settings.

    Rules summary:
    - Inputs must be dicts.
    - No unknown keys; disallow mutation of immutable keys.
    - If the project is archived, only 'archived' may be changed (to unarchive).
    - Field constraints:
        name: non-empty string, length 1..100 (trimmed)
        visibility: one of private|internal|public
        max_members: integer in [1, 100000] and >= number of members
        archived: boolean
        members: list/set/tuple of active user_ids (must exist in system), unique
        owners: list/set/tuple of active user_ids, subset of members, non-empty
        start_date/due_date: YYYY-MM-DD; start_date <= due_date if both present
        timezone: valid IANA timezone if zoneinfo available; fallback heuristic otherwise
        version: must increment by 1 relative to current version if provided
        tags: list of unique non-empty strings (<= 30 chars), at most 50 tags
        webhooks: list of unique http(s) URLs, at most 20
        sprints_enabled: boolean
        sprint_length_days: int 1..42 and only allowed when sprints_enabled is true
        allowed_roles: list of roles; all must be from known roles
    """
    if not isinstance(current_settings, dict) or not isinstance(settings_update, dict):
        return False

    # Disallow unknown keys and immutable changes
    update_keys = set(settings_update.keys())
    unknown = update_keys - _ALLOWED_KEYS - _IMMUTABLE_KEYS
    if unknown:
        return False
    if update_keys & _IMMUTABLE_KEYS:
        return False

    # If archived, only allow toggling archived
    if bool(current_settings.get("archived", False)):
        if update_keys - {"archived"}:
            return False

    # Quick per-field type checks where applicable
    if "name" in settings_update:
        name_val = settings_update["name"]
        if not _is_non_empty_str(name_val):
            return False
        if len(name_val.strip()) > 100:
            return False

    if "visibility" in settings_update:
        vis = _normalize_str(str(settings_update["visibility"]))
        if vis not in _KNOWN_VISIBILITIES:
            return False

    if "archived" in settings_update and not isinstance(settings_update["archived"], bool):
        return False

    if "max_members" in settings_update:
        mm = settings_update["max_members"]
        if not isinstance(mm, int):
            return False
        if mm < 1 or mm > 100000:
            return False

    if "timezone" in settings_update:
        if not _is_valid_timezone(settings_update["timezone"]):
            return False

    if "version" in settings_update:
        new_ver = settings_update["version"]
        if not isinstance(new_ver, int):
            return False
        curr_ver = current_settings.get("version", 0)
        if not isinstance(curr_ver, int):
            return False
        if new_ver != curr_ver + 1:
            return False

    if "tags" in settings_update:
        tags = settings_update["tags"]
        if not isinstance(tags, list):
            return False
        seen: Set[str] = set()
        if len(tags) > 50:
            return False
        for t in tags:
            if not _is_non_empty_str(t):
                return False
            tt = t.strip()
            if len(tt) > 30:
                return False
            key = tt.lower()
            if key in seen:
                return False
            seen.add(key)

    if "webhooks" in settings_update:
        hooks = settings_update["webhooks"]
        if not isinstance(hooks, list):
            return False
        if len(hooks) > 20:
            return False
        normalized_hooks: Set[str] = set()
        for h in hooks:
            if not isinstance(h, str):
                return False
            hh = h.strip()
            if not _is_http_url(hh):
                return False
            key = hh.lower()
            if key in normalized_hooks:
                return False
            normalized_hooks.add(key)

    if "sprints_enabled" in settings_update and not isinstance(settings_update["sprints_enabled"], bool):
        return False

    if "sprint_length_days" in settings_update:
        sld = settings_update["sprint_length_days"]
        if not isinstance(sld, int):
            return False
        if sld < 1 or sld > 42:
            return False

    if "allowed_roles" in settings_update:
        ar = settings_update["allowed_roles"]
        if not isinstance(ar, list):
            return False
        normalized: Set[str] = set()
        for r in ar:
            if not isinstance(r, str):
                return False
            rr = _normalize_str(r)
            if rr not in _KNOWN_ROLES:
                return False
            normalized.add(rr)
        if len(normalized) != len(ar):
            return False  # duplicates

    # Build prospective settings to validate cross-field constraints
    prospective: Dict[str, Any] = dict(current_settings)
    prospective.update(settings_update)

    # Validate dates and their relationship
    start_val = prospective.get("start_date")
    due_val = prospective.get("due_date")
    start_dt = _parse_iso_date(start_val) if start_val is not None else None
    due_dt = _parse_iso_date(due_val) if due_val is not None else None

    if start_val is not None and start_dt is None:
        return False
    if due_val is not None and due_dt is None:
        return False
    if start_dt and due_dt and start_dt > due_dt:
        return False

    # Validate members and owners
    members_raw = prospective.get("members", [])
    owners_raw = prospective.get("owners", None)

    members_set: Optional[Set[str]] = _collect_user_ids(members_raw)
    if members_set is None and "members" in prospective:
        return False
    if members_set is None:
        members_set = set()

    owners_set: Optional[Set[str]] = None
    if owners_raw is not None:
        owners_set = _collect_user_ids(owners_raw)
        if owners_set is None:
            return False

    # Owners must be a non-empty subset of members when provided or when members are present
    if owners_set is not None or "members" in prospective or "owners" in prospective:
        # Resolve owners from current if not in update
        if owners_set is None:
            owners_set = _collect_user_ids(current_settings.get("owners", [])) or set()
        # Resolve members from current if not in update
        if "members" not in prospective:
            members_set = _collect_user_ids(current_settings.get("members", [])) or set()

        if not owners_set:
            return False
        if not owners_set.issubset(members_set):
            return False

    # max_members must not be less than number of members (if defined)
    mm_value = prospective.get("max_members", None)
    if mm_value is not None:
        if not isinstance(mm_value, int):
            return False
        if len(members_set) > mm_value:
            return False

    # Sprint settings coherence
    sprints_enabled = bool(prospective.get("sprints_enabled", False))
    sld_value = prospective.get("sprint_length_days", None)
    if sld_value is not None:
        if not isinstance(sld_value, int):
            return False
        if not sprints_enabled:
            return False  # cannot set sprint length if sprints are disabled
        if sld_value < 1 or sld_value > 42:
            return False

    return True


def _normalize_settings_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize external-facing project settings into internal validation shape.
    - Map 'deadline' -> 'due_date'
    """
    if not isinstance(data, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for k, v in data.items():
        if k == "deadline":
            normalized["due_date"] = v
        else:
            normalized[k] = v
    return normalized


def _denormalize_settings_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert validated internal settings keys back to the DB storage shape.
    - Map 'due_date' -> 'deadline'
    """
    denorm: Dict[str, Any] = {}
    for k, v in data.items():
        if k == "due_date":
            denorm["deadline"] = v
        else:
            denorm[k] = v
    return denorm


def change_project_config(uid: str, role: str, proj: str, settings_update: Dict[str, Any]) -> str:
    """
    Modify project settings if the user is authorized and the settings update is valid.

    Args:
        uid: user identifier
        role: role identifier (must be possessed by the user)
        proj: project id
        settings_update: proposed changes to project settings (external shape, e.g., includes 'deadline')

    Returns:
        Success message string.

    Raises:
        KeyError: if the project does not exist
        PermissionError: if the user is not authorized
        ValueError: if the settings update payload is invalid
    """
    if not isinstance(uid, str) or not isinstance(role, str) or not isinstance(proj, str):
        raise ValueError("uid, role, and proj must be strings.")
    if not isinstance(settings_update, dict):
        raise ValueError("settings_update must be a dict.")

    # Authorization
    if not authenticate_user(uid, role):
        raise PermissionError("User is not authorized to modify project settings.")

    # Locate project
    project = PROJECTS_DB.get(proj)
    if project is None:
        raise KeyError(f"Project '{proj}' not found.")

    # Prepare current settings and the normalized update for validation
    current_settings_raw: Dict[str, Any] = project.get("settings", {})
    current_settings_norm = _normalize_settings_keys(current_settings_raw)

    update_norm = _normalize_settings_keys(settings_update)

    # Validate the proposed changes
    if not validate_project_settings(current_settings_norm, update_norm):
        raise ValueError("Invalid project settings update.")

    # Apply the update to the DB in its external/storage shape
    to_apply = _denormalize_settings_keys(update_norm)
    project.setdefault("settings", {}).update(to_apply)

    return f"Project {proj} settings updated successfully."


if __name__ == "__main__":
    # Simple manual checks (example usage):
    # print(authenticate_user("alice", "admin"))   # True
    # print(authenticate_user("alice", "manager")) # False
    # print(authenticate_user("dave", "viewer"))   # False (inactive)
    pass
