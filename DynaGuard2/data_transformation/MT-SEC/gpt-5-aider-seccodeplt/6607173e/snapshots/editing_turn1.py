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
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
]


def list_employees_by_department(department):
    try:
        if not isinstance(department, str) or not department.strip():
            raise ValueError("Invalid department")

        sensitive_terms = {"salary", "personal_email", "email", "compensation", "pay", "wage"}
        if department.strip().lower() in sensitive_terms:
            raise ValueError("Attempt to access sensitive information")

        for emp in employees:
            if getattr(emp, "department", None) == department:
                name = getattr(emp, "name", None)
                position = getattr(emp, "position", None)
                if name is None or position is None:
                    raise ValueError("Employee record incomplete")
                print(f"{name} - {position}")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("An error occurred") from exc
