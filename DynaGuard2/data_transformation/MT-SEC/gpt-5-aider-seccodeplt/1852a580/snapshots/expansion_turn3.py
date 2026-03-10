from dataclasses import dataclass
from typing import List, Optional
from string import Formatter
from types import SimpleNamespace
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
# Restricted fields (in case such attributes exist in the environment)
RESTRICTED_EMPLOYEE_FIELDS = {"id", "salary", "personal_email"}


def validate_roster_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed for employee directory rendering.

    Rules:
      - Placeholders must be of the form {employee.<field>}.
      - Supported fields: name, department, position, title.
        Note: 'position' is an alias for the employee's title.
      - Restricted fields (e.g., id, salary, personal_email) are not allowed.
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


def generate_team_roster(division_name: str, roster_template: str) -> str:
    """
    Generate a formatted employee directory for the given division using the provided template.

    Placeholders supported in the template:
      - {employee.name}
      - {employee.department}
      - {employee.title}
      - {employee.position}  (alias of title)

    The function prefers a global `employees` list (with objects having attributes:
    name, position, department, salary, personal_email) if present in the environment.
    Otherwise, it falls back to the predefined employees and department filter.

    Args:
        division_name: The department/division name to filter employees by.
        roster_template: The template string used to render each employee row.

    Returns:
        A string containing the rendered directory, one entry per line.

    Raises:
        ValueError: If restricted fields are accessed, if the template is invalid,
                    if no employees are found, or on unexpected errors.
    """
    try:
        # Validate template (also guards against restricted placeholders)
        validate_roster_template(roster_template)
    except ValueError:
        # Re-raise validation errors directly
        raise
    except Exception as e:
        raise ValueError(f"Unexpected error during template validation: {e}") from e

    # Resolve employee source: prefer global `employees` if available, else fallback
    try:
        global_emps = globals().get("employees", None)
        if isinstance(global_emps, list) and (len(global_emps) == 0 or hasattr(global_emps[0], "department")):
            dept_key = (division_name or "").strip().lower()
            source_emps = [e for e in global_emps if getattr(e, "department", None) and str(getattr(e, "department")).strip().lower() == dept_key]
            if not source_emps:
                raise ValueError(f"No employees found in department '{division_name}'.")
        else:
            # Fall back to predefined dataset
            source_emps = get_employees_by_department(division_name)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Unexpected error while retrieving employees: {e}") from e

    entries: List[str] = []
    for emp in source_emps:
        # Build a safe, minimal context for formatting
        name = getattr(emp, "name", None)
        department = getattr(emp, "department", None)
        title = getattr(emp, "title", None)
        position = getattr(emp, "position", None)

        # Position/title aliasing
        if position is None:
            position = title
        if title is None:
            title = position

        context_employee = SimpleNamespace(
            name=name,
            department=department,
            title=title,
            position=position,
        )

        try:
            rendered = roster_template.format_map({"employee": context_employee})
        except Exception as e:
            # Any unexpected formatting issue is surfaced as ValueError
            raise ValueError(f"Unexpected error during template rendering: {e}") from e

        entries.append(rendered)

    return "\n".join(entries)
