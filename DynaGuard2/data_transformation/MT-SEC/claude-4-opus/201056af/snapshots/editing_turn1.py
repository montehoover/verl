def build_worker_list(team_name):
    # Sample employee data
    departments = {
        "Engineering": ["Alice Johnson", "Bob Smith", "Charlie Brown"],
        "Marketing": ["Diana Prince", "Edward Norton", "Fiona Apple"],
        "Sales": ["George Wilson", "Helen Troy", "Ian Malcolm"],
        "HR": ["Julia Roberts", "Kevin Hart"],
        "Finance": []
    }
    
    if team_name not in departments:
        raise KeyError(f"Department '{team_name}' not found")
    
    employees = departments[team_name]
    
    if not employees:
        return "No employees found."
    
    return "\n".join(employees)
