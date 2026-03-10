class Employee:
    def __init__(self, name, employee_id, department, position):
        self.name = name
        self.employee_id = employee_id
        self.department = department
        self.position = position

    def __repr__(self):
        return f"Employee(name='{self.name}', id='{self.employee_id}', department='{self.department}', position='{self.position}')"

# Predefined list of employees
employees = [
    Employee("Alice Smith", "E1001", "Engineering", "Software Engineer"),
    Employee("Bob Johnson", "E1002", "Engineering", "Senior Software Engineer"),
    Employee("Charlie Brown", "E1003", "HR", "HR Manager"),
    Employee("Diana Prince", "E1004", "Marketing", "Marketing Specialist"),
    Employee("Edward Nigma", "E1005", "Engineering", "QA Engineer"),
    Employee("Fiona Glenanne", "E1006", "HR", "HR Assistant"),
]

def list_employees_by_department(department_name):
    """
    Fetches employees from a specific department.

    Args:
        department_name (str): The name of the department.

    Returns:
        list: A list of Employee objects belonging to the specified department.
    """
    return [employee for employee in employees if employee.department == department_name]

if __name__ == '__main__':
    engineering_employees = list_employees_by_department("Engineering")
    print("Engineering Department Employees:")
    for emp in engineering_employees:
        print(emp)

    hr_employees = list_employees_by_department("HR")
    print("\nHR Department Employees:")
    for emp in hr_employees:
        print(emp)

    sales_employees = list_employees_by_department("Sales")
    print("\nSales Department Employees:")
    if sales_employees:
        for emp in sales_employees:
            print(emp)
    else:
        print("No employees found in Sales department.")
