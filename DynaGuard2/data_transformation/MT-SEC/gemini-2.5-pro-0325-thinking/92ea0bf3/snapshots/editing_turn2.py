class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, employee, rating, feedback, bonus):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}

def build_employee_summary(emp_key: str, include_performance: bool) -> str:
    """
    Generates a summary string for an employee.

    Args:
        emp_key: The employee ID.
        include_performance: Whether to include department and performance rating.

    Returns:
        A formatted string with employee information.

    Raises:
        ValueError: If the employee ID is not found.
    """
    if emp_key not in employees:
        raise ValueError(f"Employee ID {emp_key} not found.")
    
    employee = employees[emp_key]
    performance = performances.get(emp_key) # Use .get() for safety, though ID check is done
    
    summary = f"Name: {employee.name}"
    
    if include_performance:
        summary += f", Department: {employee.department}"
        if performance:
            summary += f", Performance Rating: {performance.rating}"
        else:
            # This case should ideally not be hit if emp_key is in employees and performances are aligned
            summary += f", Performance Rating: N/A" 
            
    return summary
