class Employee:
    def __init__(self, name, department, employee_id):
        self.name = name
        self.department = department
        self.employee_id = employee_id

# Predefined list of employees
employees = [
    Employee("Alice Johnson", "Engineering", "E001"),
    Employee("Bob Smith", "Engineering", "E002"),
    Employee("Carol Williams", "Sales", "S001"),
    Employee("David Brown", "Marketing", "M001"),
    Employee("Eva Davis", "Sales", "S002"),
    Employee("Frank Miller", "Engineering", "E003"),
    Employee("Grace Wilson", "Marketing", "M002"),
    Employee("Henry Moore", "Sales", "S003"),
    Employee("Iris Taylor", "HR", "H001"),
    Employee("Jack Anderson", "HR", "H002")
]

def list_employees_by_department(department_name):
    return [emp for emp in employees if emp.department == department_name]

def validate_format_template(template):
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{employee\.\w+\}', template)
    
    # List of sensitive fields
    sensitive_fields = ['salary', 'personal_email']
    
    # List of approved fields
    approved_fields = ['name', 'position', 'department', 'employee_id']
    
    # Check for sensitive fields
    for placeholder in placeholders:
        field = placeholder.strip('{}').split('.')[1]
        if field in sensitive_fields:
            raise ValueError(f"Template contains sensitive field: {placeholder}")
    
    # Check if all placeholders are approved
    for placeholder in placeholders:
        field = placeholder.strip('{}').split('.')[1]
        if field not in approved_fields:
            return False
    
    return True
