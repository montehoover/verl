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


def create_employee_directory(department, template):
    try:
        if not isinstance(department, str) or not department.strip():
            raise ValueError("Invalid department")
        if not isinstance(template, str):
            raise ValueError("Invalid template")

        sensitive_terms = {"salary", "personal_email", "email", "compensation", "pay", "wage"}
        allowed_fields = {"name", "position", "department"}

        # Validate template fields and block access to sensitive information
        formatter = Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if not field_name:
                continue
            # Normalize tokens by splitting on '.' and '['
            tokens = []
            remainder = field_name
            # Split by '.' first
            for part in remainder.replace("[", ".").split("."):
                if not part:
                    continue
                # Strip possible closing bracket from index usage like [key]
                tokens.append(part.rstrip("]"))

            # If any token is sensitive, block it
            for token in tokens:
                if token in sensitive_terms:
                    raise ValueError("Attempt to access sensitive information")

            # Optionally enforce that fields start with 'employee' but allow others to be blank
            # No exception for non-sensitive unknown fields to keep behavior graceful

        class _SafeEmployee:
            def __init__(self, emp):
                self._emp = emp

            def __getattr__(self, attr):
                if attr in sensitive_terms:
                    raise ValueError("Attempt to access sensitive information")
                if attr in allowed_fields:
                    return getattr(self._emp, attr, "") or ""
                # Unknown/non-allowed attributes resolve to empty string
                return ""

            def __getitem__(self, key):
                # Support {employee[name]} style
                return self.__getattr__(key)

            def __str__(self):
                # If {employee} is used directly, render empty to be safe
                return ""

            def __format__(self, format_spec):
                # Ensure formatting doesn't try to format the object itself
                return ""

        class _SafeMapping(dict):
            def __missing__(self, key):
                # Any unknown top-level placeholder becomes empty string
                if key == "employee":
                    # Will be explicitly set where needed; default empty if not set
                    return _SafeEmployee(Employee("", "", "", 0, ""))
                return ""

        lines = []
        for emp in employees:
            if getattr(emp, "department", None) == department:
                mapping = _SafeMapping(employee=_SafeEmployee(emp))
                rendered = template.format_map(mapping)
                lines.append(rendered)

        return "\n".join(lines)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("An error occurred") from exc
