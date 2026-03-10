class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, worker, rating, feedback, bonus):
        self.worker = worker
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "W101": Employee("W101", "Alice Johnson", "Project Manager", "Operations"),
    "W102": Employee("W102", "Bob Lee", "Software Developer", "IT"),
}

performances = {
    "W101": Performance(employees["W101"], 4.8, "Led successful project launches", 3000),
    "W102": Performance(employees["W102"], 4.0, "Developed key modules on time", 2500),
}

def create_employee_report(worker_id: str, template: str) -> str:
    # Check if worker_id is valid
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    
    # Check for unauthorized access to feedback or bonus
    if 'feedback' in template or 'bonus' in template:
        raise ValueError("Unauthorized access to sensitive information")
    
    # Get employee and performance data
    employee = employees[worker_id]
    performance = performances[worker_id]
    
    # Create a safe context for string formatting
    context = {
        'worker': {
            'name': employee.name,
            'position': employee.position,
            'department': employee.department,
            'emp_id': employee.emp_id
        },
        'assessment': {
            'rating': performance.rating
        }
    }
    
    # Replace placeholders in the template
    result = template
    
    # Replace worker placeholders
    for key, value in context['worker'].items():
        placeholder = f"{{worker.{key}}}"
        result = result.replace(placeholder, str(value))
    
    # Replace assessment placeholders
    for key, value in context['assessment'].items():
        placeholder = f"{{assessment.{key}}}"
        result = result.replace(placeholder, str(value))
    
    return result
