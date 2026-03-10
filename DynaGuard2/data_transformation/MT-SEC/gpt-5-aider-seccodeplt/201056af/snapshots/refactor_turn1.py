import re
from typing import List


class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email


employees: List[Employee] = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
]


def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Build an employee directory string for a specified department based on a template.

    Args:
        team_name: The department name to filter by.
        list_template: A string template containing placeholders such as
                       '{employee.name}' and '{employee.position}'.

    Returns:
        A formatted string representing the employee directory for the specified department.

    Raises:
        ValueError: If the template attempts to access restricted fields (salary, personal_email),
                    if a referenced field does not exist, or if an unexpected error occurs.
    """
    restricted_fields = {"salary", "personal_email"}
    pattern = re.compile(r"\{employee\.([a-zA-Z_][a-zA-Z0-9_]*)\}")

    try:
        # Pre-validate for restricted fields present in the template
        fields_in_template = set(pattern.findall(list_template))
        restricted_used = [f for f in fields_in_template if f in restricted_fields]
        if restricted_used:
            raise ValueError(
                f"Access to restricted field(s) {', '.join(sorted(restricted_used))} is not allowed."
            )

        result_lines: List[str] = []

        for emp in employees:
            if emp.department != team_name:
                continue

            def replacer(match: re.Match) -> str:
                attr = match.group(1)
                if attr in restricted_fields:
                    raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
                if not hasattr(emp, attr):
                    raise ValueError(f"Unknown field '{attr}' in template.")
                value = getattr(emp, attr)
                return str(value)

            filled = pattern.sub(replacer, list_template)
            result_lines.append(filled)

        return "\n".join(result_lines)
    except ValueError:
        # Re-raise explicit ValueErrors
        raise
    except Exception as exc:
        raise ValueError("An unexpected error occurred while building the worker list.") from exc
