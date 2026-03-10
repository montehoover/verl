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

def retrieve_employee_data(employee_id: str) -> tuple[Employee, Performance]:
    if employee_id not in performances:
        raise ValueError("Invalid employee ID")
    
    performance = performances[employee_id]
    employee = performance.personnel
    
    return employee, performance

def validate_template(template: str) -> None:
    unauthorized_fields = ['feedback', 'bonus']
    for field in unauthorized_fields:
        if f'{{personnel.{field}}}' in template or f'{{review.{field}}}' in template:
            raise ValueError(f"Unauthorized access to {field}")

def format_template(template: str, employee: Employee, performance: Performance) -> str:
    replacements = {
        '{personnel.name}': employee.name,
        '{personnel.position}': employee.position,
        '{personnel.department}': employee.department,
        '{personnel.emp_id}': employee.emp_id,
        '{review.rating}': str(performance.rating)
    }
    
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result

def compile_employee_evaluation(employee_id: str, template: str) -> str:
    # Pipeline pattern
    employee, performance = retrieve_employee_data(employee_id)
    validate_template(template)
    return format_template(template, employee, performance)
