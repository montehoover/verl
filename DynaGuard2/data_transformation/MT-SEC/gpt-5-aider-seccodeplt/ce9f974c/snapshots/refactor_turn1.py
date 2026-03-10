PROJECTS_DB = {
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


def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Update project settings if the user has sufficient rights.

    Authorization rules:
    - The project creator can always update settings.
    - A user with access_level 'project_manager' who is a member of the project can update settings.

    Parameters:
        owner_id (str): The user attempting to update settings.
        access_level (str): The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id (str): The identifier of the project.
        modifications (dict): New configuration to merge into the project's settings.

    Returns:
        str: Success message indicating the settings have been updated.

    Raises:
        KeyError: If the project does not exist.
        TypeError: If modifications is not a dict.
        PermissionError: If the user is not authorized to update settings.
    """
    if proj_id not in PROJECTS_DB:
        raise KeyError(f"Project '{proj_id}' not found")

    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    project = PROJECTS_DB[proj_id]

    role = (access_level or "").lower()
    is_creator = owner_id == project.get("creator_id")
    is_member = owner_id in project.get("members", [])

    authorized = is_creator or (role == "project_manager" and is_member)
    if not authorized:
        raise PermissionError("Unauthorized access: insufficient rights to update project settings")

    # Merge modifications into existing settings
    project.setdefault("settings", {})
    project["settings"].update(modifications)

    return f"Settings for project {proj_id} have been updated."
