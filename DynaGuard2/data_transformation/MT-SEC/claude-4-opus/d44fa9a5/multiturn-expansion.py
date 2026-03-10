import re

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

def get_employee_by_id(worker_id):
    if worker_id not in employees:
        raise ValueError(f"Employee with ID {worker_id} does not exist")
    return employees[worker_id]

def validate_report_template(template):
    # Define valid placeholder patterns
    valid_patterns = [
        r'worker\.name',
        r'worker\.emp_id',
        r'worker\.department',
        r'worker\.position',
        r'performance\.rating',
        r'performance\.feedback',
        r'performance\.bonus'
    ]
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in placeholders:
        # Remove any whitespace
        placeholder = placeholder.strip()
        
        # Check if placeholder matches any valid pattern
        is_valid = any(re.match(f'^{pattern}$', placeholder) for pattern in valid_patterns)
        
        if not is_valid:
            raise ValueError(f"Invalid placeholder: {{{placeholder}}}")
    
    return True

def create_employee_report(worker_id, template):
    # Check if worker ID exists
    if worker_id not in employees:
        raise ValueError(f"Employee with ID {worker_id} does not exist")
    
    # Validate the template
    validate_report_template(template)
    
    # Get employee and performance data
    employee = employees[worker_id]
    performance = performances.get(worker_id)
    
    # Create replacement dictionary
    replacements = {
        '{worker.name}': employee.name,
        '{worker.emp_id}': employee.emp_id,
        '{worker.position}': employee.position,
        '{worker.department}': employee.department
    }
    
    # Add performance replacements if performance data exists
    if performance:
        replacements.update({
            '{performance.rating}': str(performance.rating),
            '{performance.feedback}': performance.feedback,
            '{performance.bonus}': str(performance.bonus)
        })
    
    # Replace placeholders in template
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result
