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

def list_employees_by_department(department_name):
    try:
        department_employees = []
        for employee in employees:
            if employee.department == department_name:
                department_employees.append(employee.name)
        return department_employees
    except AttributeError as e:
        if 'salary' in str(e) or 'personal_email' in str(e):
            raise ValueError("Attempted to access restricted fields")
        else:
            raise ValueError("An unexpected error occurred")
    except Exception:
        raise ValueError("An unexpected error occurred")
