class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

def generate_employee_summary(employee_id, include_department):
    if employee_id not in employees:
        raise ValueError(f"Employee ID {employee_id} does not exist in the database")
    
    employee = employees[employee_id]
    
    if include_department:
        return f"{employee.name} - {employee.position} ({employee.department})"
    else:
        return f"{employee.name} - {employee.position}"
