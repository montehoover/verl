from typing import Any, Dict, List, Tuple

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


def adjust_project_params(
    user_token: str,
    role: str,
    project_ident: str,
    filter_params: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Retrieves project details according to role-based authorization and optional filtering.

    Parameters:
    - user_token (str): Identifier for the user (treated as user_id).
    - role (str): One of 'project_manager', 'team_member', or 'viewer'.
    - project_ident (str): Project ID to access.
    - filter_params (dict): Nested dict specifying which fields to return.
        Example filter:
            {
                "project_id": True,
                "settings": {
                    "visibility": True
                }
            }

    Authorization rules:
    - project_manager: must be a member or the creator; full access to project fields.
    - team_member: must be a member or the creator; partial access to settings only.
    - viewer: access only if project visibility is 'public'; basic info only.

    Raises:
    - KeyError if the project does not exist.
    - ValueError for invalid role.
    - PermissionError for unauthorized access or disallowed field requests.

    Returns:
    - dict with the allowed and requested project fields.
    """
    project = PROJECTS_DB.get(project_ident)
    if project is None:
        raise KeyError(f"Project '{project_ident}' not found")

    visibility = (project.get("settings") or {}).get("visibility")

    # Define allowed schemas per role
    manager_allowed: Dict[str, Any] = {
        "project_id": True,
        "creator_id": True,
        "members": True,
        "settings": {
            "visibility": True,
            "deadline": True
        }
    }
    team_allowed: Dict[str, Any] = {
        "project_id": True,
        "settings": {
            # Partial access: can view only visibility, not deadline
            "visibility": True
        }
    }
    viewer_allowed: Dict[str, Any] = {
        "project_id": True,
        "settings": {
            "visibility": True
        }
    }

    role = role.strip().lower()
    if role == "project_manager":
        if not _is_member(user_token, project):
            raise PermissionError("User is not authorized as a project manager for this project")
        allowed_schema = manager_allowed
    elif role == "team_member":
        if not _is_member(user_token, project):
            raise PermissionError("User is not a member of this project")
        allowed_schema = team_allowed
    elif role == "viewer":
        if visibility != "public":
            raise PermissionError("Viewer access denied: project is not public")
        allowed_schema = viewer_allowed
    else:
        raise ValueError("Invalid role. Expected 'project_manager', 'team_member', or 'viewer'.")

    # If no filter provided, return maximum allowed for role
    if not filter_params:
        return _extract_fields(project, allowed_schema)

    # Validate requested fields are within allowed schema
    unauthorized = _diff_schema(filter_params, allowed_schema)
    if unauthorized:
        paths = ", ".join(unauthorized)
        raise PermissionError(f"Requested fields are not permitted for role '{role}': {paths}")

    # Build effective extraction schema and return filtered data
    effective_schema = _merge_subtree_request_with_allowed(filter_params, allowed_schema)
    return _extract_fields(project, effective_schema)
