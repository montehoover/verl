def build_team_directory(dept_name):
    # Example employee data - in a real application this might come from a database
    employees = {
        "Engineering": ["Alice Johnson", "Bob Smith", "Charlie Brown"],
        "Sales": ["David Wilson", "Emma Davis", "Frank Miller"],
        "Marketing": ["Grace Lee", "Henry Taylor", "Iris Chen"],
        "HR": ["Jack Anderson", "Karen White", "Leo Martinez"]
    }
    
    if dept_name not in employees or not employees[dept_name]:
        raise ValueError(f"No employees found in department: {dept_name}")
    
    return "\n".join(employees[dept_name])
