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
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    
    employee = employees[worker_id]
    performance = performances.get(worker_id)
    
    # Check for unauthorized access
    if 'feedback' in template or 'bonus' in template:
        raise ValueError("Unauthorized access to feedback or bonus information")
    
    # Replace placeholders
    result = template
    result = result.replace('{worker.name}', employee.name)
    result = result.replace('{worker.position}', employee.position)
    result = result.replace('{worker.department}', employee.department)
    result = result.replace('{worker.emp_id}', employee.emp_id)
    
    if performance:
        result = result.replace('{assessment.rating}', str(performance.rating))
        result = result.replace('{assessment.worker.name}', performance.worker.name)
        result = result.replace('{assessment.worker.position}', performance.worker.position)
        result = result.replace('{assessment.worker.department}', performance.worker.department)
        result = result.replace('{assessment.worker.emp_id}', performance.worker.emp_id)
    
    return result
