class Employee:
    def __init__(self, name, employee_id, department):
        self.name = name
        self.employee_id = employee_id
        self.department = department

# Predefined list of employees
employees = [
    Employee("John Smith", "E001", "Engineering"),
    Employee("Sarah Johnson", "E002", "Engineering"),
    Employee("Michael Brown", "E003", "Sales"),
    Employee("Emily Davis", "E004", "HR"),
    Employee("David Wilson", "E005", "Sales"),
    Employee("Jennifer Lee", "E006", "Marketing"),
    Employee("Robert Taylor", "E007", "Engineering"),
    Employee("Lisa Anderson", "E008", "HR"),
    Employee("James Martinez", "E009", "Marketing"),
    Employee("Mary Thomas", "E010", "Sales")
]

def get_employees_by_department(department_name):
    """
    Filter employees by their department.
    
    Args:
        department_name (str): The name of the department to filter by
        
    Returns:
        list: A list of Employee objects in the specified department
        
    Raises:
        ValueError: If no employees are found in the specified department
    """
    filtered_employees = [emp for emp in employees if emp.department == department_name]
    
    if not filtered_employees:
        raise ValueError(f"No employees found in the {department_name} department")
    
    return filtered_employees
