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

def build_employee_summary(emp_key, include_performance):
    if emp_key not in employees:
        raise ValueError(f"Employee ID {emp_key} not found")
    
    employee = employees[emp_key]
    
    if include_performance and emp_key in performances:
        performance = performances[emp_key]
        return f"{employee.name} - {employee.department} - Rating: {performance.rating}"
    else:
        return f"{employee.name} - {employee.department}"
