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


def verify_user_token(user_token: str) -> bool:
    """
    Verify if a user token is valid.
    
    Args:
        user_token: The token string to verify
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    # Basic validation checks
    if not user_token:
        return False
    
    # Check token length (typical JWT tokens are quite long)
    if len(user_token) < 20:
        return False
    
    # Check for basic token structure (simplified check)
    # Real implementation would decode and verify JWT signature
    parts = user_token.split('.')
    if len(parts) != 3:
        return False
    
    # Check each part has content
    for part in parts:
        if not part:
            return False
    
    # In a real implementation, you would:
    # 1. Decode the JWT token
    # 2. Verify the signature
    # 3. Check expiration time
    # 4. Validate issuer and audience
    # 5. Check against revoked tokens
    
    # For now, basic validation passes
    return True


def check_user_role(user_id: str, project_id: str) -> str:
    """
    Check the user's role within a specific project.
    
    Args:
        user_id: The unique identifier of the user
        project_id: The unique identifier of the project
        
    Returns:
        str: The user's role in the project (e.g., 'owner', 'admin', 'member', 'viewer', 'none')
    """
    # Validate inputs
    if not user_id or not project_id:
        return 'none'
    
    # Mock database/storage of user-project relationships
    # In production, this would query a database
    user_project_roles = {
        ('user_001', 'proj_101'): 'owner',
        ('user_001', 'proj_102'): 'admin',
        ('user_002', 'proj_101'): 'member',
        ('user_002', 'proj_102'): 'viewer',
        ('user_003', 'proj_101'): 'admin',
        ('user_003', 'proj_103'): 'owner',
        ('user_004', 'proj_102'): 'member',
    }
    
    # Look up the user's role for the specific project
    role = user_project_roles.get((user_id, project_id), 'none')
    
    return role


def adjust_project_params(user_token: str, role: str, project_ident: str, updated_values: dict) -> str:
    """
    Update project settings if the user has sufficient permissions.
    
    Args:
        user_token: The user's authentication token
        role: The user's role in the project
        project_ident: The project identifier
        updated_values: Dictionary of settings to update
        
    Returns:
        str: Success message if authorized
        
    Raises:
        PermissionError: If user lacks sufficient permissions
    """
    # Verify user token first
    if not verify_user_token(user_token):
        raise PermissionError("Invalid user token")
    
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        raise PermissionError("Project not found")
    
    # Define roles that can modify project settings
    # Only owner and admin roles can update project parameters
    authorized_roles = ['owner', 'admin']
    
    if role not in authorized_roles:
        raise PermissionError(f"Insufficient permissions. Role '{role}' cannot modify project settings")
    
    # Update the project settings
    project = PROJECTS_DB[project_ident]
    
    # Update each setting in updated_values
    for key, value in updated_values.items():
        if key in project['settings']:
            project['settings'][key] = value
        else:
            # Add new setting if it doesn't exist
            project['settings'][key] = value
    
    return f"Project {project_ident} settings updated successfully"
