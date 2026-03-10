import re
from typing import List, Set


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


# Constants for template parsing and validation
RESTRICTED_FIELDS: Set[str] = {"salary", "personal_email"}
PLACEHOLDER_PATTERN = re.compile(r"\{employee\.([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _extract_template_fields(list_template: str) -> Set[str]:
    """Extract all employee attribute names referenced by the template."""
    return set(PLACEHOLDER_PATTERN.findall(list_template))


def _validate_no_restricted_fields(fields: Set[str]) -> None:
    """Ensure the template does not reference restricted fields."""
    restricted_used = sorted(RESTRICTED_FIELDS.intersection(fields))
    if restricted_used:
        raise ValueError(
            f"Access to restricted field(s) {', '.join(restricted_used)} is not allowed."
        )


def _filter_employees_by_department(team_name: str, roster: List[Employee]) -> List[Employee]:
    """Return employees whose department matches the provided team name."""
    return [emp for emp in roster if emp.department == team_name]


def _render_template_for_employee(list_template: str, emp: Employee) -> str:
    """Fill the template for a single employee, enforcing field restrictions."""

    def replacer(match: re.Match) -> str:
        attr = match.group(1)
        if attr in RESTRICTED_FIELDS:
            raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
        if not hasattr(emp, attr):
            raise ValueError(f"Unknown field '{attr}' in template.")
        value = getattr(emp, attr)
        return str(value)

    return PLACEHOLDER_PATTERN.sub(replacer, list_template)


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
    try:
        # Parse and validate template fields
        fields_in_template = _extract_template_fields(list_template)
        _validate_no_restricted_fields(fields_in_template)

        # Filter employees
        filtered = _filter_employees_by_department(team_name, employees)

        # Render each employee line
        result_lines: List[str] = [
            _render_template_for_employee(list_template, emp) for emp in filtered
        ]

        return "\n".join(result_lines)
    except ValueError:
        # Re-raise explicit ValueErrors
        raise
    except Exception as exc:
        raise ValueError("An unexpected error occurred while building the worker list.") from exc
