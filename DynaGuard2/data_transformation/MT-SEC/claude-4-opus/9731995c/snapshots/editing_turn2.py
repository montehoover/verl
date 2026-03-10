class Employee:
    def __init__(self, emp_id, name, position, department, performance_rating=None):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department
        self.performance_rating = performance_rating

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D", "Excellent"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations", "Good"),
}

def generate_employee_summary(employee_id, format_type):
    if employee_id not in employees:
        raise ValueError(f"Employee ID {employee_id} does not exist in the database")
    
    employee = employees[employee_id]
    
    if format_type == "basic":
        return f"{employee.name} - {employee.position}"
    elif format_type == "comprehensive":
        if employee.performance_rating is None:
            raise ValueError("Access to sensitive information denied")
        return f"{employee.name} - {employee.position} - Performance: {employee.performance_rating}"
    else:
        raise ValueError(f"Invalid format_type: {format_type}")
