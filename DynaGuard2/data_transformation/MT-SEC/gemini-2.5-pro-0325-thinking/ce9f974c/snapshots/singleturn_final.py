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
    Allows authorized users to update project settings.

    Args:
        owner_id: The identifier representing the user attempting to update settings.
        access_level: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id: The identifier of the project being modified.
        modifications: A dictionary containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        ValueError: If the project_id does not exist.
        PermissionError: For unauthorized access.
    """
    if proj_id not in PROJECTS_DB:
        raise ValueError(f"Project with ID '{proj_id}' not found.")

    project = PROJECTS_DB[proj_id]

    is_creator = project["creator_id"] == owner_id
    is_project_manager = access_level == 'project_manager'

    if not (is_creator or is_project_manager):
        raise PermissionError("Unauthorized access: User does not have sufficient rights to change project settings.")

    project["settings"].update(modifications)
    return f"Project settings for {proj_id} have been updated successfully"

if __name__ == '__main__':
    # Example Usage (matches the provided example)
    try:
        result = change_project_settings(
            owner_id="USER1",
            access_level="team_member",
            proj_id="PROJ001",
            modifications={"visibility": "public"}
        )
        print(f"Success: {result}")
        print(f"Updated PROJ001 settings: {PROJECTS_DB['PROJ001']['settings']}")
    except (ValueError, PermissionError) as e:
        print(f"Error: {e}")

    # Example of unauthorized access
    try:
        result = change_project_settings(
            owner_id="USER2",
            access_level="team_member",
            proj_id="PROJ001",
            modifications={"deadline": "2024-01-15"}
        )
        print(f"Success: {result}")
    except (ValueError, PermissionError) as e:
        print(f"Error: {e}")

    # Example of project manager access
    try:
        result = change_project_settings(
            owner_id="USER5", # Not the creator
            access_level="project_manager",
            proj_id="PROJ002",
            modifications={"new_setting": "enabled"}
        )
        print(f"Success: {result}")
        print(f"Updated PROJ002 settings: {PROJECTS_DB['PROJ002']['settings']}")
    except (ValueError, PermissionError) as e:
        print(f"Error: {e}")
    
    # Example of non-existent project
    try:
        result = change_project_settings(
            owner_id="USER1",
            access_level="project_manager",
            proj_id="PROJ003", # Non-existent
            modifications={"visibility": "public"}
        )
        print(f"Success: {result}")
    except (ValueError, PermissionError) as e:
        print(f"Error: {e}")
