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
    # Check if employee ID is valid
    if employee_id not in employees:
        raise ValueError(f"Invalid employee ID: {employee_id}")
    
    # Check for unauthorized access to feedback or bonus
    if 'feedback' in template.lower() or 'bonus' in template.lower():
        raise ValueError("Unauthorized access to sensitive information")
    
    # Get employee and performance data
    employee = employees[employee_id]
    performance = performances[employee_id]
    
    # Create a safe context for string formatting
    context = {
        'personnel': {
            'name': employee.name,
            'position': employee.position,
            'department': employee.department,
            'emp_id': employee.emp_id
        },
        'review': {
            'rating': performance.rating
        }
    }
    
    # Replace placeholders in the template
    result = template
    
    # Replace personnel placeholders
    for key, value in context['personnel'].items():
        placeholder = f"{{personnel.{key}}}"
        result = result.replace(placeholder, str(value))
    
    # Replace review placeholders
    for key, value in context['review'].items():
        placeholder = f"{{review.{key}}}"
        result = result.replace(placeholder, str(value))
    
    return result
