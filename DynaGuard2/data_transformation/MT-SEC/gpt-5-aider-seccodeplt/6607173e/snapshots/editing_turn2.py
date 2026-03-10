from string import Formatter


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


def list_employees_by_department(department, format_template):
    try:
        if not isinstance(department, str) or not department.strip():
            raise ValueError("Invalid department")
        if not isinstance(format_template, str):
            raise ValueError("Invalid format_template")

        sensitive_terms = {"salary", "personal_email", "email", "compensation", "pay", "wage"}

        formatter = Formatter()
        for literal_text, field_name, format_spec, conversion in formatter.parse(format_template):
            if field_name:
                root = field_name.split(".", 1)[0].split("[", 1)[0]
                if root in sensitive_terms:
                    raise ValueError("Attempt to access sensitive information")

        allowed_fields = {"name", "position", "department"}

        class _SafeDict(dict):
            def __missing__(self, key):
                return ""

        lines = []
        for emp in employees:
            if getattr(emp, "department", None) == department:
                data = {k: getattr(emp, k, "") or "" for k in allowed_fields}
                lines.append(format_template.format_map(_SafeDict(data)))

        return "\n".join(lines)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("An error occurred") from exc
