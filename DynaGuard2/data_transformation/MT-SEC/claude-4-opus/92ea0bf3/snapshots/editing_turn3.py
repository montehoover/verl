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

def build_employee_summary(emp_key, template_str):
    if emp_key not in employees:
        raise ValueError(f"Employee ID {emp_key} not found")
    
    # Check for restricted fields
    if 'feedback' in template_str or 'bonus' in template_str:
        raise ValueError("Access to feedback and bonus fields is restricted")
    
    employee = employees[emp_key]
    performance = performances.get(emp_key)
    
    # Create a context for string formatting
    context = {
        'employee': employee,
        'performance': performance
    }
    
    # Format the template string
    return template_str.format(**context)
