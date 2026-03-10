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


def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Build a formatted directory string for employees in the specified department.

    Args:
        dept_name: The department name to filter employees by.
        format_template: The format string template to apply per employee.

    Returns:
        A newline-separated string with one formatted line per employee.

    Raises:
        ValueError: If the template is non-compliant/sensitive or formatting fails.
    """
    # Basic argument validation
    if not isinstance(dept_name, str) or not isinstance(format_template, str):
        raise ValueError("Invalid arguments")

    # Validate the template for approved placeholders and sensitive access
    is_valid = validate_format_template(format_template)
    if not is_valid:
        raise ValueError("Template contains unapproved placeholders")

    dept = dept_name.strip()
    if not dept:
        return ""

    # Prefer a global 'employees' list if available, otherwise fall back to EMPLOYEES
    people = globals().get("employees", None)
    if not isinstance(people, list):
        people = EMPLOYEES

    # Filter by department
    try:
        team = [p for p in people if getattr(p, "department", None) == dept]
    except Exception as e:
        raise ValueError("Error filtering employees by department") from e

    # Format each line; any formatting issue results in a ValueError
    lines: List[str] = []
    for person in team:
        try:
            line = format_template.format(employee=person)
        except Exception as e:
            raise ValueError("Error formatting employee entry") from e
        lines.append(line)

    return "\n".join(lines)
