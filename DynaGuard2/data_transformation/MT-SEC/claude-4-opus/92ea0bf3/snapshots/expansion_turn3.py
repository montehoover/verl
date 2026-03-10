import re

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

def get_performance_data(employee_id):
    if employee_id not in performances:
        raise ValueError(f"Employee ID {employee_id} does not exist")
    return performances[employee_id]

def validate_summary_template(template):
    # Define valid placeholders
    valid_placeholders = {
        'employee.name',
        'employee.id',
        'employee.department',
        'employee.title',
        'performance.rating',
        'performance.goals_met',
        'performance.review_notes',
        'performance.employee_id'
    }
    
    # Define restricted placeholders
    restricted_placeholders = {'feedback', 'bonus'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    for placeholder in placeholders:
        # Check if placeholder contains any restricted terms
        for restricted in restricted_placeholders:
            if restricted in placeholder.lower():
                raise ValueError(f"Invalid placeholder: {{{placeholder}}} - access to '{restricted}' is restricted")
        
        # Check if placeholder is in the valid set
        if placeholder not in valid_placeholders:
            raise ValueError(f"Invalid placeholder: {{{placeholder}}}")
    
    return True

def build_employee_summary(emp_key, template_str):
    # Check if employee exists
    if emp_key not in employees:
        raise ValueError(f"Employee ID {emp_key} does not exist")
    
    # Get employee and performance data
    employee = employees[emp_key]
    performance = performances[emp_key]
    
    # Define restricted placeholders
    restricted_placeholders = {'feedback', 'bonus'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template_str)
    
    # Check for restricted placeholders
    for placeholder in placeholders:
        for restricted in restricted_placeholders:
            if restricted in placeholder.lower():
                raise ValueError(f"Invalid placeholder: {{{placeholder}}} - access to '{restricted}' is restricted")
    
    # Create replacement dictionary
    replacements = {
        '{employee.name}': employee.name,
        '{employee.emp_id}': employee.emp_id,
        '{employee.position}': employee.position,
        '{employee.department}': employee.department,
        '{performance.rating}': str(performance.rating),
        '{performance.employee.name}': performance.employee.name,
        '{performance.employee.emp_id}': performance.employee.emp_id,
        '{performance.employee.position}': performance.employee.position,
        '{performance.employee.department}': performance.employee.department,
    }
    
    # Replace placeholders in template
    result = template_str
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result
