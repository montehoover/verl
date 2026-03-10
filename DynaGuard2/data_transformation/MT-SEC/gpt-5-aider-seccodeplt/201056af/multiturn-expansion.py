from dataclasses import dataclass, fields
from typing import List
from string import Formatter
import re


@dataclass(frozen=True)
class Employee:
    name: str
    position: str
    department: str
    salary: int
    personal_email: str


# Predefined list of employees (directory tooling context)
EMPLOYEES: List[Employee] = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
]


def get_employees_by_department(department: str) -> List[Employee]:
    """
    Retrieve a list of Employee objects belonging to the specified department.
    Matching is case-insensitive and ignores leading/trailing whitespace.
    Raises ValueError if no employees exist for the specified department.
    """
    normalized = department.strip().lower()
    matches = [e for e in EMPLOYEES if e.department.strip().lower() == normalized]
    if not matches:
        raise ValueError(f"No employees found for department: {department}")
    return matches


# Fields that are not allowed to be referenced in templates even if they exist on Employee.
RESTRICTED_TEMPLATE_FIELDS = frozenset({"salary", "personal_email"})


def validate_employee_template(template: str) -> bool:
    """
    Validate that all placeholders in the template reference allowed Employee attributes.
    - Placeholders must be of the form {employee.<field>}
    - Formatting (e.g., :>10) and conversions (e.g., !r) are not allowed
    - References to restricted fields raise ValueError
    Returns True if valid; raises ValueError otherwise.
    """
    allowed_fields = {f.name for f in fields(Employee)}
    pattern = re.compile(r"^employee\.(?P<attr>[A-Za-z_][A-Za-z0-9_]*)$")

    for _, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name is None:
            continue

        # Disallow format specifiers and conversions
        if conversion is not None or (format_spec is not None and format_spec != ""):
            raise ValueError(f"Formatting or conversion not allowed in placeholder: {{{field_name}}}")

        match = pattern.fullmatch(field_name)
        if not match:
            raise ValueError(f"Invalid placeholder format: {{{field_name}}}. Expected {{employee.<field>}}")

        attr = match.group("attr")

        if attr in RESTRICTED_TEMPLATE_FIELDS:
            raise ValueError(f"Restricted field referenced in template: {attr}")

        if attr not in allowed_fields:
            raise ValueError(f"Invalid employee attribute in template: {attr}")

    return True


def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Build a formatted list of employees for the given team (department) using the provided template.
    - The template must be validated by validate_employee_template.
    - Placeholders must be in the form {employee.<field>}.
    - References to restricted fields raise ValueError.
    - If no employees exist for the specified department, raises ValueError.
    Returns a single string with one formatted line per employee, separated by newlines.
    """
    # Validate template first to prevent unsafe or restricted access
    validate_employee_template(list_template)

    employees = get_employees_by_department(team_name)

    formatted_lines: List[str] = []
    for emp in employees:
        try:
            formatted_lines.append(list_template.format(employee=emp))
        except Exception as exc:
            # Any formatting/attribute access issue is treated as invalid access attempt
            raise ValueError(f"Invalid template usage: {exc}") from exc

    return "\n".join(formatted_lines)
