class Employee:
    def __init__(self, name, employee_id, department):
        self.name = name
        self.employee_id = employee_id
        self.department = department
    
    def __repr__(self):
        return f"Employee(name='{self.name}', id={self.employee_id}, dept='{self.department}')"

# Predefined list of employees
employees = [
    Employee("Alice Johnson", 101, "Engineering"),
    Employee("Bob Smith", 102, "Engineering"),
    Employee("Carol Williams", 103, "Marketing"),
    Employee("David Brown", 104, "Sales"),
    Employee("Emma Davis", 105, "Marketing"),
    Employee("Frank Wilson", 106, "HR"),
    Employee("Grace Lee", 107, "Engineering"),
    Employee("Henry Martin", 108, "Sales"),
    Employee("Iris Chen", 109, "HR"),
    Employee("Jack Taylor", 110, "Finance")
]

def get_employees_by_department(department_name):
    """
    Filter employees by department name.
    
    Args:
        department_name (str): The name of the department to filter by
        
    Returns:
        list: List of Employee objects in the specified department
        
    Raises:
        ValueError: If the department does not exist
    """
    # Get all unique departments
    all_departments = set(emp.department for emp in employees)
    
    # Check if the department exists
    if department_name not in all_departments:
        raise ValueError(f"Department '{department_name}' does not exist. Available departments: {', '.join(sorted(all_departments))}")
    
    # Filter employees by department
    return [emp for emp in employees if emp.department == department_name]
