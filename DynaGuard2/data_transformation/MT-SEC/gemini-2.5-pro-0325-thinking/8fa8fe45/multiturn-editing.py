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

def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    """
    Modifies the settings for a project.

    Args:
        member_id: The ID of the user trying to modify the settings.
        member_role: The user's role (e.g., 'admin', 'editor', 'viewer').
        proj_id: The ID of the project.
        fresh_settings: A dictionary containing the updated settings.

    Returns:
        A success message if the settings are updated, otherwise an error message.
    """
    project = PROJECTS_DB.get(proj_id)

    if not project:
        return f"Error: Project with ID '{proj_id}' not found."

    # Authorization check: User must be the project creator or have an 'admin' role.
    if project["creator_id"] == member_id or member_role == "admin":
        project["settings"].update(fresh_settings)
        return f"Success: Settings for project '{proj_id}' updated."
    else:
        return f"Error: User '{member_id}' is not authorized to change settings for project '{proj_id}'."

if __name__ == '__main__':
    # Example Usage for alter_project_preferences

    # Scenario 1: Project creator updates settings
    result1 = alter_project_preferences(
        member_id="USER1",
        member_role="creator",
        proj_id="PROJ001",
        fresh_settings={"visibility": "public", "new_setting": "value1"}
    )
    print(result1)
    print(f"PROJ001 new settings: {PROJECTS_DB.get('PROJ001', {}).get('settings')}")

    # Scenario 2: Admin updates settings for a project they didn't create
    result2 = alter_project_preferences(
        member_id="ADMIN_USER", # An admin user not necessarily the creator
        member_role="admin",
        proj_id="PROJ002",
        fresh_settings={"deadline": "2024-01-31", "status": "active"}
    )
    print(result2)
    print(f"PROJ002 new settings: {PROJECTS_DB.get('PROJ002', {}).get('settings')}")

    # Scenario 3: Unauthorized user (not creator, not admin) tries to update settings
    result3 = alter_project_preferences(
        member_id="USER2", # Member but not creator or admin
        member_role="editor",
        proj_id="PROJ001",
        fresh_settings={"visibility": "restricted"}
    )
    print(result3)
    print(f"PROJ001 settings (should be unchanged by USER2): {PROJECTS_DB.get('PROJ001', {}).get('settings')}")

    # Scenario 4: Attempt to update settings for a non-existent project
    result4 = alter_project_preferences(
        member_id="USER1",
        member_role="creator",
        proj_id="PROJ003", # Non-existent project
        fresh_settings={"visibility": "private"}
    )
    print(result4)
