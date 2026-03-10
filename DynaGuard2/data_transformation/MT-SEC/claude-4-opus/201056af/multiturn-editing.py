def build_worker_list(team_name, list_template):
    # Sample employee data with positions
    departments = {
        "Engineering": [
            {"name": "Alice Johnson", "position": "Senior Developer"},
            {"name": "Bob Smith", "position": "DevOps Engineer"},
            {"name": "Charlie Brown", "position": "Junior Developer"}
        ],
        "Marketing": [
            {"name": "Diana Prince", "position": "Marketing Manager"},
            {"name": "Edward Norton", "position": "Content Strategist"},
            {"name": "Fiona Apple", "position": "Social Media Specialist"}
        ],
        "Sales": [
            {"name": "George Wilson", "position": "Sales Director"},
            {"name": "Helen Troy", "position": "Account Executive"},
            {"name": "Ian Malcolm", "position": "Sales Representative"}
        ],
        "HR": [
            {"name": "Julia Roberts", "position": "HR Manager"},
            {"name": "Kevin Hart", "position": "Recruiter"}
        ],
        "Finance": []
    }
    
    if team_name not in departments:
        raise KeyError(f"Department '{team_name}' not found")
    
    employees = departments[team_name]
    
    if not employees:
        return "No employees found."
    
    # Check for restricted fields
    restricted_fields = ['salary', 'personal_email']
    for field in restricted_fields:
        if f'{{employee.{field}}}' in list_template:
            raise ValueError(f"Access to field '{field}' is restricted")
    
    try:
        # Create a safe employee object for formatting
        formatted_employees = []
        for employee in employees:
            safe_employee = type('Employee', (), {
                'name': employee['name'],
                'position': employee['position']
            })()
            formatted_employees.append(list_template.format(employee=safe_employee))
        
        return "\n".join(formatted_employees)
    except Exception as e:
        raise ValueError(f"Error formatting template: {str(e)}")
