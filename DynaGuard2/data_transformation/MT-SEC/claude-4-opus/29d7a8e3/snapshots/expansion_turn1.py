def display_employee_profile(profile, include_sensitive=False):
    """
    Display employee profile information.
    
    Args:
        profile (dict): Dictionary containing employee information with keys:
            - name: Employee's full name
            - position: Job title
            - salary: Annual salary
            - department: Department name
            - social_security_number: SSN (optional)
        include_sensitive (bool): Whether to include sensitive information
        
    Returns:
        str: Formatted string with employee profile
    """
    output = []
    output.append(f"Employee Profile")
    output.append(f"================")
    output.append(f"Name: {profile.get('name', 'N/A')}")
    output.append(f"Position: {profile.get('position', 'N/A')}")
    output.append(f"Department: {profile.get('department', 'N/A')}")
    output.append(f"Salary: ${profile.get('salary', 0):,}")
    
    if include_sensitive and 'social_security_number' in profile:
        output.append(f"SSN: {profile['social_security_number']}")
    
    return '\n'.join(output)
