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

def list_employees_by_department(department_name, output_template='{name}'):
    try:
        department_employees = []
        for employee in employees:
            if employee.department == department_name:
                formatted_output = output_template
                if '{name}' in formatted_output:
                    formatted_output = formatted_output.replace('{name}', employee.name)
                if '{position}' in formatted_output:
                    formatted_output = formatted_output.replace('{position}', employee.position)
                if '{department}' in formatted_output:
                    formatted_output = formatted_output.replace('{department}', employee.department)
                if '{salary}' in formatted_output:
                    raise ValueError("Attempted to access restricted fields")
                if '{personal_email}' in formatted_output:
                    raise ValueError("Attempted to access restricted fields")
                department_employees.append(formatted_output)
        return department_employees
    except AttributeError as e:
        if 'salary' in str(e) or 'personal_email' in str(e):
            raise ValueError("Attempted to access restricted fields")
        else:
            raise ValueError("An unexpected error occurred")
    except Exception as e:
        if "Attempted to access restricted fields" in str(e):
            raise
        raise ValueError("An unexpected error occurred")
