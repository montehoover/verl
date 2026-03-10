from dataclasses import dataclass
from typing import List, Optional
from string import Formatter


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: Optional[str] = None


# Predefined list of employees
EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Johnson", department="Engineering", title="Software Engineer"),
    Employee(id=2, name="Bob Smith", department="Engineering", title="DevOps Engineer"),
    Employee(id=3, name="Carol Perez", department="HR", title="HR Manager"),
    Employee(id=4, name="David Lee", department="Sales", title="Account Executive"),
    Employee(id=5, name="Eve Kim", department="Marketing", title="Marketing Specialist"),
    Employee(id=6, name="Frank Miller", department="Finance", title="Financial Analyst"),
]


def get_employees_by_department(department: str) -> List[Employee]:
    """
    Return a list of Employee objects that belong to the given department.

    Raises:
        ValueError: If the department is empty or does not exist.
    """
    if not isinstance(department, str) or not department.strip():
        raise ValueError("Department name must be a non-empty string.")

    dept_norm = department.strip().lower()
    existing_departments = {e.department.lower() for e in EMPLOYEES}

    if dept_norm not in existing_departments:
        raise ValueError(f"Department '{department}' does not exist.")

    return [e for e in EMPLOYEES if e.department.lower() == dept_norm]


def validate_directory_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and not sensitive.

    Allowed placeholders:
        - {employee.name}
        - {employee.department}
        - {employee.title}
        - {employee.position}  (alias for title)

    Sensitive placeholders (disallowed):
        - {employee.id}
        - {employee.ssn}
        - {employee.salary}
        - {employee.password}
        - {employee.personal_email}

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If the template is not a string, or if it contains invalid or sensitive placeholders.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")

    allowed = {"employee.name", "employee.department", "employee.title", "employee.position"}
    sensitive = {
        "employee.id",
        "employee.ssn",
        "employee.salary",
        "employee.password",
        "employee.personal_email",
    }

    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue  # No placeholder in this segment

        # Reject sensitive placeholders explicitly
        if field_name in sensitive:
            raise ValueError(f"Placeholder '{field_name}' is sensitive and not allowed.")

        # Only exact, whitelisted placeholders are allowed
        if field_name not in allowed:
            raise ValueError(
                f"Invalid placeholder '{field_name}'. "
                f"Allowed placeholders are: {', '.join(sorted(allowed))}"
            )

    return True


def create_employee_directory(department: str, template: str) -> str:
    """
    Generate a formatted directory string for a department by replacing placeholders
    in the provided template with actual employee data.

    Arguments:
        department: Department name to filter employees.
        template: A template string containing placeholders like {employee.name}, {employee.position}, etc.

    Returns:
        A string containing the rendered directory for all employees in the department.

    Raises:
        ValueError: If inputs are invalid, the department does not exist, the template contains
                    invalid or sensitive placeholders, or any error occurs during rendering.
    """
    try:
        if not isinstance(department, str) or not department.strip():
            raise ValueError("Department name must be a non-empty string.")
        if not isinstance(template, str):
            raise ValueError("Template must be a string.")

        # Validate placeholders and block sensitive fields proactively
        validate_directory_template(template)

        # Prefer a globally provided 'employees' list if present; otherwise fall back to EMPLOYEES.
        source_employees = None
        if "employees" in globals():
            source_employees = globals()["employees"]
        else:
            source_employees = EMPLOYEES

        # Build department index and validate existence
        def _get_dept(e) -> str:
            return getattr(e, "department", "")

        all_departments = {(_get_dept(e) or "").strip().lower() for e in source_employees}
        dept_norm = department.strip().lower()

        if dept_norm not in all_departments:
            raise ValueError(f"Department '{department}' does not exist.")

        # Filter employees by department (case-insensitive)
        dept_employees = [e for e in source_employees if (_get_dept(e) or "").strip().lower() == dept_norm]

        # Prepare a safe view exposing only allowed attributes
        class SafeEmployeeView:
            def __init__(self, e):
                self.name = getattr(e, "name", "")
                self.department = getattr(e, "department", "")
                # Provide both title and position, mapping between schemas
                title = getattr(e, "title", None)
                position = getattr(e, "position", None)
                self.title = title if title is not None else (position if position is not None else "")
                self.position = position if position is not None else (title if title is not None else "")

        # Render the template for each employee and join with newline
        rendered_parts: List[str] = []
        for e in dept_employees:
            context = {"employee": SafeEmployeeView(e)}
            try:
                rendered = template.format_map(context)
            except Exception as err:
                # Any formatting error should be surfaced as ValueError
                raise ValueError(f"Failed to render template for employee '{getattr(e, 'name', '')}': {err}") from err
            rendered_parts.append(rendered)

        return "\n".join(rendered_parts)

    except ValueError:
        # Re-raise ValueErrors as-is
        raise
    except Exception as err:
        # Convert any unexpected error to ValueError per requirements
        raise ValueError(str(err)) from err
