def build_worker_list(team_name):
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
    
    return "\n".join([f"Name: {employee['name']}, Position: {employee['position']}" for employee in employees])
