class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, emp_id, rating, feedback, bonus):
        self.emp_id = emp_id
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance("E101", "Excellent", "Outstanding technical skills", 15000),
    "E102": Performance("E102", "Good", "Strong analytical abilities", 10000),
}

def generate_employee_summary(employee_id, template):
    if employee_id not in employees:
        raise ValueError(f"Employee ID {employee_id} does not exist in the database")
    
    if "feedback" in template or "bonus" in template:
        raise ValueError("Access to sensitive information denied")
    
    employee = employees[employee_id]
    performance = performances.get(employee_id)
    
    # Replace employee placeholders
    result = template.replace("{employee.name}", employee.name)
    result = result.replace("{employee.position}", employee.position)
    result = result.replace("{employee.department}", employee.department)
    
    # Replace performance placeholders
    if performance:
        result = result.replace("{performance.rating}", performance.rating)
    
    return result
