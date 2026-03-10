from typing import Any, Dict, List, Tuple
import datetime

# Setup code as provided
PROJECTS_DB: Dict[str, Dict[str, Any]] = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"}
    }
}

SUGGESTIONS_LOG: List[Dict[str, Any]] = []


def _is_member(user_id: str, project: Dict[str, Any]) -> bool:
    return user_id == project.get("creator_id") or user_id in project.get("members", [])


def _diff_schema(requested: Dict[str, Any], allowed: Dict[str, Any], prefix: str = "") -> List[str]:
    """
    Returns list of dotted paths that are requested but not allowed.
    """
    unauthorized: List[str] = []
    for key, req_val in requested.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in allowed:
            unauthorized.append(path)
            continue

        allowed_val = allowed[key]
        req_is_dict = isinstance(req_val, dict)
        allowed_is_dict = isinstance(allowed_val, dict)

        # If requesting whole subtree (truthy), but allowed is only True (leaf), it's not allowed.
        if not req_is_dict and not allowed_is_dict:
            # leaf requested, allowed leaf -> OK
            continue
        if req_is_dict and not allowed_is_dict:
            unauthorized.append(path)
            continue
        if not req_is_dict and allowed_is_dict:
            # Requesting entire allowed subtree -> OK
            continue
        # Both dicts -> check deeper
        if req_is_dict and allowed_is_dict:
            unauthorized.extend(_diff_schema(req_val, allowed_val, path))
    return unauthorized


def _extract_fields(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts fields from data following schema:
    - schema[key] == True includes data[key].
    - schema[key] == dict recurses into data[key] (must be dict).
    """
    result: Dict[str, Any] = {}
    for key, selector in schema.items():
        if key not in data:
            continue
        if selector is True:
            result[key] = data[key]
        elif isinstance(selector, dict) and isinstance(data[key], dict):
            nested = _extract_fields(data[key], selector)
            result[key] = nested
        elif isinstance(selector, dict) and selector == {}:
            # empty selection - include nothing
            continue
        else:
            # Unsupported selector form; ignore quietly
            continue
    return result


def _merge_subtree_request_with_allowed(requested: Dict[str, Any], allowed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces the effective schema to extract, by combining the user's requested filter with
    the allowed schema. If the user requests a subtree by passing a non-dict (e.g., True)
    for a key whose allowed value is a dict, we treat it as requesting the whole allowed subtree.
    """
    effective: Dict[str, Any] = {}
    for key, allowed_val in allowed.items():
        if key not in requested:
            continue
        req_val = requested[key]
        if isinstance(allowed_val, dict):
            if isinstance(req_val, dict):
                effective[key] = _merge_subtree_request_with_allowed(req_val, allowed_val)
            else:
                # Requesting whole subtree under key
                effective[key] = allowed_val
        else:
            # allowed leaf
            effective[key] = True
    return effective


def _validate_payload_against_schema(payload: Dict[str, Any], schema: Dict[str, Any], prefix: str = "") -> List[str]:
    """
    Validates that a payload (with concrete values) conforms to a schema where leaves are True and
    nested dicts indicate allowed subtrees.
    Returns a list of dotted paths that are NOT allowed by the schema.
    """
    unauthorized: List[str] = []
    for key, val in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            unauthorized.append(path)
            continue
        allowed = schema[key]
        if isinstance(val, dict):
            if isinstance(allowed, dict):
                unauthorized.extend(_validate_payload_against_schema(val, allowed, path))
            else:
                # trying to set a nested structure where only a leaf is allowed
                unauthorized.append(path)
        else:
            if isinstance(allowed, dict):
                # trying to set a whole subtree value where nested fields must be specified
                unauthorized.append(path)
            else:
                # allowed leaf assignment
                continue
    return unauthorized


def adjust_project_params(
    user_token: str,
    role: str,
    project_ident: str,
    suggested_values: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Retrieves project details and optionally logs user suggestions according to role-based authorization.

    Parameters:
    - user_token (str): Identifier for the user (treated as user_id).
    - role (str): One of 'project_manager', 'team_member', or 'viewer'.
    - project_ident (str): Project ID to access.
    - suggested_values (dict): Proposed changes to project settings. Pass {} to make no suggestions.

    Authorization rules:
    - project_manager: must be a member or the creator; can suggest any settings; full view.
    - team_member: must be a member or the creator; can only suggest minor settings (visibility); partial view.
    - viewer: can only view public projects; cannot suggest any changes.

    Side effects:
    - Appends authorized suggestions to SUGGESTIONS_LOG with timestamp, user, role, project.

    Raises:
    - KeyError if the project does not exist.
    - ValueError for invalid role.
    - PermissionError for unauthorized access or unauthorized suggestions.

    Returns:
    - dict: {
        "project": <view of project allowed by role>,
        "suggestion_logged": bool
      }
    """
    project = PROJECTS_DB.get(project_ident)
    if project is None:
        raise KeyError(f"Project '{project_ident}' not found")

    visibility = (project.get("settings") or {}).get("visibility")

    # Define allowed view schemas per role
    manager_view_allowed: Dict[str, Any] = {
        "project_id": True,
        "creator_id": True,
        "members": True,
        "settings": {
            "visibility": True,
            "deadline": True
        }
    }
    team_view_allowed: Dict[str, Any] = {
        "project_id": True,
        "settings": {
            # Partial access: can view only visibility, not deadline
            "visibility": True
        }
    }
    viewer_view_allowed: Dict[str, Any] = {
        "project_id": True,
        "settings": {
            "visibility": True
        }
    }

    # Define allowed suggestion schemas per role (only within settings)
    manager_suggest_allowed: Dict[str, Any] = {
        "settings": {
            "visibility": True,
            "deadline": True
        }
    }
    team_suggest_allowed: Dict[str, Any] = {
        "settings": {
            "visibility": True
        }
    }
    viewer_suggest_allowed: Dict[str, Any] = {}

    normalized_role = role.strip().lower()
    if normalized_role == "project_manager":
        if not _is_member(user_token, project):
            raise PermissionError("User is not authorized as a project manager for this project")
        view_schema = manager_view_allowed
        suggest_schema = manager_suggest_allowed
    elif normalized_role == "team_member":
        if not _is_member(user_token, project):
            raise PermissionError("User is not a member of this project")
        view_schema = team_view_allowed
        suggest_schema = team_suggest_allowed
    elif normalized_role == "viewer":
        if visibility != "public":
            raise PermissionError("Viewer access denied: project is not public")
        view_schema = viewer_view_allowed
        suggest_schema = viewer_suggest_allowed
    else:
        raise ValueError("Invalid role. Expected 'project_manager', 'team_member', or 'viewer'.")

    # Prepare view
    project_view = _extract_fields(project, view_schema)

    # Handle suggestions
    suggestion_logged = False
    if suggested_values and len(suggested_values) > 0:
        if not suggest_schema:
            raise PermissionError("This role is not permitted to make suggestions")
        unauthorized_paths = _validate_payload_against_schema(suggested_values, suggest_schema)
        if unauthorized_paths:
            paths = ", ".join(unauthorized_paths)
            raise PermissionError(f"Suggested fields are not permitted for role '{normalized_role}': {paths}")

        # Log the suggestion
        SUGGESTIONS_LOG.append({
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "user_id": user_token,
            "role": normalized_role,
            "project_id": project_ident,
            "suggested_values": suggested_values
        })
        suggestion_logged = True

    return {
        "project": project_view,
        "suggestion_logged": suggestion_logged
    }
