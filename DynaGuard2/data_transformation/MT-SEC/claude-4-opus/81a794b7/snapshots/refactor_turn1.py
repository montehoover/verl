class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}

def compile_employee_evaluation(employee_id: str, template: str) -> str:
    if employee_id not in performances:
        raise ValueError("Invalid employee ID")
    
    performance = performances[employee_id]
    employee = performance.personnel
    
    # Check for unauthorized fields
    unauthorized_fields = ['feedback', 'bonus']
    for field in unauthorized_fields:
        if f'{{personnel.{field}}}' in template or f'{{review.{field}}}' in template:
            raise ValueError(f"Unauthorized access to {field}")
    
    # Replace placeholders
    result = template
    result = result.replace('{personnel.name}', employee.name)
    result = result.replace('{personnel.position}', employee.position)
    result = result.replace('{personnel.department}', employee.department)
    result = result.replace('{personnel.emp_id}', employee.emp_id)
    result = result.replace('{review.rating}', str(performance.rating))
    
    return result
