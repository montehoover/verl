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


def _safe_getattr(obj, attr):
    if attr in {"salary", "personal_email"}:
        raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
    return getattr(obj, attr)


def list_employees_by_department(department_name, output_template):
    """
    Return a list of formatted strings for employees in the given department_name.
    The output_template may include placeholders like '{name}', '{position}', and '{department}'.
    Missing placeholders are replaced with empty strings.

    Raises:
        ValueError: if restricted fields are accessed or if any unexpected error occurs.
    """
    try:
        if output_template is None:
            output_template = "{name}"

        # Detect any attempt to access restricted fields via placeholders
        formatter = Formatter()
        restricted = {"salary", "personal_email"}
        for _, field_name, _, _ in formatter.parse(output_template):
            if field_name and field_name in restricted:
                raise ValueError(f"Access to restricted field '{field_name}' is not allowed.")

        result = []
        for emp in employees:
            if _safe_getattr(emp, "department") == department_name:
                context = {
                    "name": _safe_getattr(emp, "name"),
                    "position": _safe_getattr(emp, "position"),
                    "department": _safe_getattr(emp, "department"),
                }

                class _SafeDefault(dict):
                    def __missing__(self, key):
                        # Gracefully handle missing placeholders
                        return ""

                formatted = output_template.format_map(_SafeDefault(context))
                result.append(formatted)

        return result
    except ValueError:
        # Re-raise restricted field access violations as-is
        raise
    except Exception as exc:
        # Wrap any other unexpected errors
        raise ValueError(f"An unexpected error occurred: {exc}")


class _SafeMissing:
    def __getattr__(self, _):
        return self

    def __getitem__(self, _):
        return self

    def __str__(self):
        return ""

    def __format__(self, _):
        return ""


class _SafeDict(dict):
    def __missing__(self, key):
        return _SafeMissing()


class _SafeEmployeeProxy:
    _restricted = {"salary", "personal_email"}

    def __init__(self, employee):
        self._employee = employee

    def __getattr__(self, attr):
        if attr in self._restricted:
            raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
        if hasattr(self._employee, attr):
            return getattr(self._employee, attr)
        return _SafeMissing()


def generate_team_roster(division_name, roster_template):
    """
    Return a formatted string representing the employee directory for the specified department.
    The roster_template may include placeholders like '{employee.name}', '{employee.position}', or '{employee.department}'.
    Missing placeholders or attributes are replaced with empty strings.

    Raises:
        ValueError: if restricted fields are accessed or if any unexpected error occurs.
    """
    try:
        if roster_template is None:
            roster_template = "{employee.name}"

        # Detect any attempt to access restricted fields via placeholders (e.g., '{employee.salary}')
        formatter = Formatter()
        restricted = {"salary", "personal_email"}
        for _, field_name, _, _ in formatter.parse(roster_template):
            if not field_name:
                continue
            parts = field_name.split(".")
            if parts and parts[0] == "employee":
                # Check any attribute segment after 'employee'
                for seg in parts[1:]:
                    if seg in restricted:
                        raise ValueError(f"Access to restricted field '{seg}' is not allowed.")

        lines = []
        for emp in employees:
            if _safe_getattr(emp, "department") == division_name:
                context = _SafeDict(employee=_SafeEmployeeProxy(emp))
                rendered = roster_template.format_map(context)
                lines.append(rendered)

        return "\n".join(lines)
    except ValueError:
        # Re-raise restricted field access violations as-is
        raise
    except Exception as exc:
        # Wrap any other unexpected errors
        raise ValueError(f"An unexpected error occurred: {exc}")
