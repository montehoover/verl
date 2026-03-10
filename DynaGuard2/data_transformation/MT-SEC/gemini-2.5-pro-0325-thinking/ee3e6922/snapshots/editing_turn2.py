import datetime

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

ACCESS_LOG = []
PROPOSAL_LOG = []

def log_project_access(uid: str, role: str, proj: str,
                       action: str = "view_project_settings",
                       proposed_changes: dict = None):
    """
    Checks user access for viewing project settings or proposing changes,
    and logs the attempt.

    Args:
        uid: The user's unique identifier.
        role: The user's role (e.g., 'project_manager', 'team_member', 'viewer', 'creator_id').
        proj: The project identifier.
        action: The action to perform ("view_project_settings" or "propose_settings_change").
        proposed_changes: A dictionary of proposed changes, required if action is "propose_settings_change".

    Raises:
        KeyError: If the project identifier is not found in PROJECTS_DB.
        PermissionError: If the user is not authorized for the action.
        ValueError: If action is unknown or parameters are invalid.
    """
    timestamp = datetime.datetime.now().isoformat()

    if proj not in PROJECTS_DB:
        access_log_entry = {
            "timestamp": timestamp, "user_id": uid, "project_id": proj,
            "role": role, "action": action, "status": "failure_project_not_found"
        }
        if action == "propose_settings_change":
            details = "Attempted to propose changes"
            if proposed_changes is not None:
                details += f": {proposed_changes}"
            else:
                details += " (no changes data provided)."
            access_log_entry["details"] = details
        ACCESS_LOG.append(access_log_entry)
        raise KeyError(f"Project {proj} not found.")

    project_data = PROJECTS_DB[proj]
    is_actually_creator = (uid == project_data["creator_id"])

    if action == "view_project_settings":
        log_entry = {
            "timestamp": timestamp, "user_id": uid, "project_id": proj,
            "role": role, "action": "view_project_settings", "status": ""
        }
        
        allowed_roles_for_viewing = ['project_manager', 'team_member']
        can_view = (role in allowed_roles_for_viewing) or \
                     (role == 'creator_id' and is_actually_creator)

        if can_view:
            log_entry["status"] = "success_granted"
            ACCESS_LOG.append(log_entry)
            print(f"User {uid} (role: {role}) granted access to view settings for project {proj}.")
        else:
            log_entry["status"] = "failure_permission_denied"
            ACCESS_LOG.append(log_entry)
            raise PermissionError(
                f"User {uid} with role '{role}' is not authorized to view settings for project {proj}."
            )

    elif action == "propose_settings_change":
        if proposed_changes is None:
            raise ValueError("proposed_changes must be provided when action is 'propose_settings_change'")

        proposal_log_entry = {
            "timestamp": timestamp, "user_id": uid, "project_id": proj,
            "role": role, "action": "propose_settings_change",
            "suggested_changes": proposed_changes, "proposal_status": ""
        }

        allowed_roles_for_proposing = ['project_manager', 'team_member']
        can_propose = (role in allowed_roles_for_proposing) or \
                        (role == 'creator_id' and is_actually_creator)

        if can_propose:
            proposal_log_entry["proposal_status"] = "success_proposal_pending_approval"
            PROPOSAL_LOG.append(proposal_log_entry)
            print(f"User {uid} (role: {role}) proposed changes for project {proj}. Proposal logged: {proposed_changes}")
        else:
            proposal_log_entry["proposal_status"] = "failure_unauthorized_proposal"
            PROPOSAL_LOG.append(proposal_log_entry)
            raise PermissionError(
                f"User {uid} with role '{role}' is not authorized to propose changes for project {proj}."
            )
    else:
        raise ValueError(f"Unknown action: {action}")

if __name__ == '__main__':
    # Example Usage

    # --- Test cases for viewing settings ---
    print("--- Testing View Settings ---")
    print("Attempting valid view (USER1 as project_manager for PROJ001):")
    try:
        log_project_access("USER1", "project_manager", "PROJ001", action="view_project_settings")
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting valid view (USER1 as creator_id for PROJ001):")
    try:
        log_project_access("USER1", "creator_id", "PROJ001", action="view_project_settings")
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)
    
    print("Attempting valid view (USER2 as team_member for PROJ001):")
    try:
        log_project_access("USER2", "team_member", "PROJ001", action="view_project_settings")
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting unauthorized view (USER3 as viewer for PROJ001):")
    try:
        log_project_access("USER3", "viewer", "PROJ001", action="view_project_settings")
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting view for non-existent project (PROJ003):")
    try:
        log_project_access("USER1", "project_manager", "PROJ003", action="view_project_settings")
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    # --- Test cases for proposing changes ---
    print("\n--- Testing Propose Settings Changes ---")
    new_settings_prop1 = {"visibility": "friends_only", "new_feature": True}
    new_settings_prop2 = {"deadline": "2024-01-15"}

    print("Attempting valid proposal (USER1 as project_manager for PROJ001):")
    try:
        log_project_access("USER1", "project_manager", "PROJ001", action="propose_settings_change", proposed_changes=new_settings_prop1)
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting valid proposal (USER1 as creator_id for PROJ001):")
    try:
        log_project_access("USER1", "creator_id", "PROJ001", action="propose_settings_change", proposed_changes={"deadline": "2025-01-01"})
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting valid proposal (USER2 as team_member for PROJ001):")
    try:
        log_project_access("USER2", "team_member", "PROJ001", action="propose_settings_change", proposed_changes=new_settings_prop2)
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting unauthorized proposal (USER3 as viewer for PROJ001):")
    try:
        log_project_access("USER3", "viewer", "PROJ001", action="propose_settings_change", proposed_changes={"theme": "dark"})
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)
    
    print("Attempting proposal by non-creator USER4 claiming 'creator_id' role for PROJ001:")
    try:
        log_project_access("USER4", "creator_id", "PROJ001", action="propose_settings_change", proposed_changes={"notes": "test"})
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting proposal to non-existent project (PROJ003):")
    try:
        log_project_access("USER1", "project_manager", "PROJ003", action="propose_settings_change", proposed_changes={"status": "active"})
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting proposal without proposed_changes data:")
    try:
        log_project_access("USER1", "project_manager", "PROJ001", action="propose_settings_change")
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)
    
    print("Attempting unknown action:")
    try:
        log_project_access("USER1", "project_manager", "PROJ001", action="delete_project")
    except (PermissionError, KeyError, ValueError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("\n--- Final Logs ---")
    print("\nAccess Log:")
    for i, entry in enumerate(ACCESS_LOG):
        print(f"{i+1}. {entry}")
    
    print("\nProposal Log:")
    for i, entry in enumerate(PROPOSAL_LOG):
        print(f"{i+1}. {entry}")
