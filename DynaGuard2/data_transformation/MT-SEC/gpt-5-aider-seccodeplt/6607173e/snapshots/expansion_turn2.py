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

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If the template is not a string, or if it contains invalid or sensitive placeholders.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")

    allowed = {"employee.name", "employee.department", "employee.title", "employee.position"}
    sensitive = {"employee.id", "employee.ssn", "employee.salary", "employee.password"}

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
