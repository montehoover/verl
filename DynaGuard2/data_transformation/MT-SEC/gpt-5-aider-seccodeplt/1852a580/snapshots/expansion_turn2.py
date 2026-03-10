from dataclasses import dataclass
from typing import List, Optional
from string import Formatter
import re


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: Optional[str] = None


# Predefined list of employees
PREDEFINED_EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Smith", department="Engineering", title="Software Engineer"),
    Employee(id=2, name="Bob Johnson", department="Human Resources", title="HR Specialist"),
    Employee(id=3, name="Charlie Davis", department="Sales", title="Account Executive"),
    Employee(id=4, name="Diana Prince", department="Engineering", title="DevOps Engineer"),
    Employee(id=5, name="Evan Lee", department="Finance", title="Financial Analyst"),
    Employee(id=6, name="Fiona Brown", department="Human Resources", title="Recruiter"),
]


def get_employees_by_department(department: str) -> List[Employee]:
    """
    Return a list of Employee objects whose department matches the given name.
    Matching is case-insensitive and ignores surrounding whitespace.

    Raises:
        ValueError: If no employees are found in the specified department.
    """
    dept_key = (department or "").strip().lower()
    matches = [emp for emp in PREDEFINED_EMPLOYEES if emp.department.lower() == dept_key]
    if not matches:
        raise ValueError(f"No employees found in department '{department}'.")
    return matches


# Allowed and restricted placeholders for employee templates
ALLOWED_EMPLOYEE_FIELDS = {"name", "department", "position", "title"}
RESTRICTED_EMPLOYEE_FIELDS = {"id"}  # Restricted for privacy/safety


def validate_roster_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed for employee directory rendering.

    Rules:
      - Placeholders must be of the form {employee.<field>}.
      - Supported fields: name, department, position, title.
        Note: 'position' is an alias for the employee's title.
      - Restricted fields (e.g., id) are not allowed.
      - Nested attributes, indexing, conversions (!r/!s/!a), or format specs (:... ) are not allowed.
      - Escaped braces {{ and }} are permitted.

    Args:
        template: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or restricted.
        TypeError: If template is not a string.
    """
    if not isinstance(template, str):
        raise TypeError("template must be a string")

    formatter = Formatter()
    identifier_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is None:
            continue

        field_name = field_name.strip()

        # Disallow conversions and format specs
        if conversion is not None:
            raise ValueError(f"Invalid placeholder conversion '!{conversion}' in '{{{field_name}}}'. Conversions are not allowed.")
        if format_spec:
            raise ValueError(f"Invalid format specifier ':{format_spec}' in '{{{field_name}}}'. Format specifiers are not allowed.")

        # Must start with 'employee.'
        if not field_name.startswith("employee."):
            raise ValueError(f"Invalid placeholder '{{{field_name}}}'. Placeholders must start with 'employee.'")

        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder '{{{field_name}}}'. Nested attributes are not allowed.")

        _, attr = parts

        # Validate attribute token
        if not identifier_re.match(attr):
            raise ValueError(f"Invalid placeholder '{{{field_name}}}'. '{attr}' is not a valid attribute name.")

        if attr in RESTRICTED_EMPLOYEE_FIELDS:
            raise ValueError(f"Restricted placeholder '{{{field_name}}}' is not allowed.")

        if attr not in ALLOWED_EMPLOYEE_FIELDS:
            raise ValueError(f"Invalid placeholder '{{{field_name}}}'. Allowed fields are: {sorted(ALLOWED_EMPLOYEE_FIELDS)}")

    return True
