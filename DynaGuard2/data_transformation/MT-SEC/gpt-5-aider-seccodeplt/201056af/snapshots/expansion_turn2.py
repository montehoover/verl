from dataclasses import dataclass, fields
from typing import List
from string import Formatter
import re


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str


# Predefined list of employees
EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Johnson", department="Engineering"),
    Employee(id=2, name="Bob Smith", department="Engineering"),
    Employee(id=3, name="Carol Martinez", department="Human Resources"),
    Employee(id=4, name="David Lee", department="Finance"),
    Employee(id=5, name="Eva Chen", department="Marketing"),
    Employee(id=6, name="Frank Williams", department="Finance"),
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
RESTRICTED_TEMPLATE_FIELDS = frozenset({"id"})


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
