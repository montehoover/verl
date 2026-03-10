from dataclasses import dataclass
from typing import List
from string import Formatter


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: str


# Predefined list of employees
EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Johnson", department="Engineering", title="Software Engineer"),
    Employee(id=2, name="Bob Smith", department="Engineering", title="DevOps Engineer"),
    Employee(id=3, name="Carol Lee", department="HR", title="HR Manager"),
    Employee(id=4, name="David Kim", department="Sales", title="Account Executive"),
    Employee(id=5, name="Eve Torres", department="HR", title="Recruiter"),
]


def list_employees_by_department(department_name: str) -> List[Employee]:
    """
    Return a list of Employee objects that belong to the given department.
    Uses the predefined EMPLOYEES list.

    Args:
        department_name: The exact name of the department to filter by.

    Returns:
        A list of Employee objects in the specified department.
    """
    if not isinstance(department_name, str):
        return []
    dept = department_name.strip()
    if not dept:
        return []
    return [emp for emp in EMPLOYEES if emp.department == dept]


def validate_format_template(template: str) -> bool:
    """
    Validate that a format string template uses only approved placeholders.

    Approved placeholders include:
      - {employee.name}
      - {employee.position}
      - {employee.title}
      - {employee.department}
      - {employee.id}

    Sensitive placeholders that must not appear (will raise ValueError):
      - {employee.salary}
      - {employee.personal_email}

    Any placeholder not explicitly approved will cause this function to return False.

    Args:
        template: The format template string to validate.

    Returns:
        True if the template is compliant; False if it contains non-approved placeholders.

    Raises:
        ValueError: If the template contains sensitive placeholders.
    """
    if not isinstance(template, str):
        return False

    approved_placeholders = {
        "employee.name",
        "employee.position",
        "employee.title",
        "employee.department",
        "employee.id",
    }
    sensitive_indicators = {
        "employee.salary",
        "employee.personal_email",
    }

    formatter = Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(template):
        # Skip literals without fields
        if field_name is None:
            continue

        # Normalize field name (ignore surrounding whitespace)
        field = field_name.strip()

        # If any sensitive indicator is referenced anywhere in the field, block immediately
        for sensitive in sensitive_indicators:
            if sensitive in field:
                raise ValueError(f"Template contains sensitive placeholder: {{{sensitive}}}")

        # Only allow exact approved placeholders; anything else is non-compliant
        if field not in approved_placeholders:
            return False

        # Optional: format spec and conversion are allowed but don't affect approval of the field
        _ = format_spec, conversion  # explicitly unused

    return True
