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


def _safe_getattr(obj, attr):
    if attr in {"salary", "personal_email"}:
        raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
    return getattr(obj, attr)


def list_employees_by_department(department_name):
    """
    Return a list of employee names belonging to the given department_name.
    Raises:
        ValueError: if restricted fields are accessed or if any unexpected error occurs.
    """
    try:
        result = []
        for emp in employees:
            if _safe_getattr(emp, "department") == department_name:
                result.append(_safe_getattr(emp, "name"))
        return result
    except ValueError:
        # Re-raise restricted field access violations as-is
        raise
    except Exception as exc:
        # Wrap any other unexpected errors
        raise ValueError(f"An unexpected error occurred: {exc}")
