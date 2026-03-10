class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]

def list_employees_by_department(department):
    try:
        found = False
        for employee in employees:
            if employee.department == department:
                print(f"Name: {employee.name}, Position: {employee.position}")
                found = True
        
        if not found:
            print(f"No employees found in department: {department}")
            
    except Exception as e:
        raise ValueError(f"An error occurred while listing employees: {str(e)}")
