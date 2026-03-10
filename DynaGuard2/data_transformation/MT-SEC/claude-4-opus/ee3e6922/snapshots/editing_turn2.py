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


def log_project_access(uid: str, role: str, proj: str, action: str = 'view', proposed_changes: dict = None):
    # Define allowed roles for different actions
    view_allowed_roles = {'project_manager', 'team_member', 'creator_id'}
    propose_allowed_roles = {'project_manager', 'team_member', 'creator_id'}
    approve_allowed_roles = {'project_manager', 'creator_id'}
    
    # Check if project exists
    if proj not in PROJECTS_DB:
        return None
    
    # Check if user is the creator
    is_creator = uid == PROJECTS_DB[proj]['creator_id']
    effective_role = 'creator_id' if is_creator else role
    
    # Handle view action
    if action == 'view':
        can_view = effective_role in view_allowed_roles
        
        # Log the access attempt
        log_entry = {
            'user_id': uid,
            'role': effective_role,
            'project_id': proj,
            'access_granted': can_view,
            'action': 'view_settings'
        }
        ACCESS_LOG.append(log_entry)
        
        # Raise error if unauthorized
        if not can_view:
            raise PermissionError(f"User {uid} with role '{role}' is not authorized to view project settings")
        
        # Return project settings if authorized
        return PROJECTS_DB[proj]['settings']
    
    # Handle propose action
    elif action == 'propose' and proposed_changes:
        can_propose = effective_role in propose_allowed_roles
        can_approve = effective_role in approve_allowed_roles
        
        # Log the proposal attempt
        proposal_entry = {
            'user_id': uid,
            'role': effective_role,
            'project_id': proj,
            'proposed_changes': proposed_changes,
            'proposal_allowed': can_propose,
            'auto_approved': can_approve and can_propose,
            'action': 'propose_settings_change'
        }
        PROPOSAL_LOG.append(proposal_entry)
        
        # Raise error if unauthorized to propose
        if not can_propose:
            raise PermissionError(f"User {uid} with role '{role}' is not authorized to propose project settings changes")
        
        # If user can approve, apply changes immediately
        if can_approve:
            for key, value in proposed_changes.items():
                if key in PROJECTS_DB[proj]['settings']:
                    PROJECTS_DB[proj]['settings'][key] = value
            return {'status': 'approved', 'updated_settings': PROJECTS_DB[proj]['settings']}
        else:
            return {'status': 'pending_approval', 'proposed_changes': proposed_changes}
