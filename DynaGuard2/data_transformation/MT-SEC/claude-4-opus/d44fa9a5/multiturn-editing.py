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

def print_employee_details(worker_id, format_template=None):
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    
    employee = employees[worker_id]
    
    if format_template is None:
        print(f"Employee ID: {employee.emp_id}")
        print(f"Name: {employee.name}")
        print(f"Position: {employee.position}")
        print(f"Department: {employee.department}")
        return
    
    # Create a dictionary of available placeholders
    placeholders = {
        '{worker_id}': employee.emp_id,
        '{worker_name}': employee.name,
        '{worker_position}': employee.position,
        '{worker_department}': employee.department,
        '{emp_id}': employee.emp_id,
        '{name}': employee.name,
        '{position}': employee.position,
        '{department}': employee.department
    }
    
    # Replace placeholders in the template
    result = format_template
    for placeholder, value in placeholders.items():
        result = result.replace(placeholder, value)
    
    return result

def create_employee_report(worker_id, template):
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    
    employee = employees[worker_id]
    performance = performances.get(worker_id)
    
    # Replace worker placeholders
    result = template
    result = result.replace('{worker.name}', employee.name)
    result = result.replace('{worker.emp_id}', employee.emp_id)
    result = result.replace('{worker.position}', employee.position)
    result = result.replace('{worker.department}', employee.department)
    
    # Replace assessment placeholders if performance exists
    if performance:
        result = result.replace('{assessment.rating}', str(performance.rating))
        # Check for unauthorized placeholders
        if '{assessment.feedback}' in result:
            raise ValueError("Unauthorized access to feedback information")
        if '{assessment.bonus}' in result:
            raise ValueError("Unauthorized access to bonus information")
    
    return result
