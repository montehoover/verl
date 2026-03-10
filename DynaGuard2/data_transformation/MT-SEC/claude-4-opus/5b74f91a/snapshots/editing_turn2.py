def build_team_directory(dept_name):
    # Example employee data - in a real application this might come from a database
    employees = {
        "Engineering": [
            ("Alice Johnson", "Senior Software Engineer"),
            ("Bob Smith", "DevOps Engineer"),
            ("Charlie Brown", "Junior Developer")
        ],
        "Sales": [
            ("David Wilson", "Sales Director"),
            ("Emma Davis", "Account Executive"),
            ("Frank Miller", "Sales Representative")
        ],
        "Marketing": [
            ("Grace Lee", "Marketing Manager"),
            ("Henry Taylor", "Content Strategist"),
            ("Iris Chen", "Digital Marketing Specialist")
        ],
        "HR": [
            ("Jack Anderson", "HR Director"),
            ("Karen White", "Recruiter"),
            ("Leo Martinez", "HR Coordinator")
        ]
    }
    
    if dept_name not in employees or not employees[dept_name]:
        raise ValueError(f"No employees found in department: {dept_name}")
    
    result = []
    for name, role in employees[dept_name]:
        result.append(f"{name}\n{role}")
    
    return "\n\n".join(result)
