import re
from dataclasses import dataclass
from typing import Dict, Union, Set


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: str
    email: str


# Predefined dictionary of employees
EMPLOYEES: Dict[int, Employee] = {
    1001: Employee(id=1001, name="Alice Johnson", department="Engineering", title="Software Engineer", email="alice.johnson@example.com"),
    1002: Employee(id=1002, name="Bob Smith", department="Marketing", title="Marketing Manager", email="bob.smith@example.com"),
    1003: Employee(id=1003, name="Carol Davis", department="Human Resources", title="HR Specialist", email="carol.davis@example.com"),
}


def get_employee_by_id(employee_id: Union[int, str]) -> Employee:
    """
    Retrieve an Employee by ID.

    Args:
        employee_id: The ID of the employee (int or numeric string).

    Returns:
        The corresponding Employee object.

    Raises:
        ValueError: If the ID is invalid or does not exist.
    """
    try:
        emp_id = int(employee_id)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid employee ID: {employee_id!r}")

    try:
        return EMPLOYEES[emp_id]
    except KeyError:
        raise ValueError(f"Employee ID {emp_id} does not exist")


# Allowed placeholders for template validation
# Using the runtime-provided Employee and Performance structures for validation:
# - personnel: Employee attributes -> emp_id, name, position, department
# - review: Performance attributes -> rating, feedback, bonus
_ALLOWED_PERSONNEL_FIELDS: Set[str] = {"emp_id", "name", "position", "department"}
_ALLOWED_REVIEW_FIELDS: Set[str] = {"rating", "feedback", "bonus"}

ALLOWED_PLACEHOLDERS: Dict[str, Set[str]] = {
    "personnel": _ALLOWED_PERSONNEL_FIELDS,
    "review": _ALLOWED_REVIEW_FIELDS,
}

_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_template_placeholders(template: str) -> bool:
    """
    Validate that all placeholders in the template are authorized and well-formed.

    Placeholders must be in the form {namespace.field}, for example:
      - {personnel.name}
      - {review.rating}

    Valid namespaces:
      - personnel: emp_id, name, position, department
      - review: rating, feedback, bonus

    Args:
        template: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If the template is not a string, or if any placeholder is
                    malformed or unauthorized.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")

    for match in _PLACEHOLDER_PATTERN.finditer(template):
        token = match.group(1).strip()
        if not token:
            raise ValueError("Empty placeholder {} is not allowed")

        parts = token.split(".")
        if len(parts) != 2:
            raise ValueError(f"Placeholder must be in the form '{{namespace.field}}': {{{token}}}")

        namespace, field = parts[0].strip(), parts[1].strip()

        if not _NAME_PATTERN.match(namespace) or not _NAME_PATTERN.match(field):
            raise ValueError(f"Invalid placeholder name syntax: {{{token}}}")

        if namespace not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Unauthorized placeholder namespace '{namespace}' in {{{token}}}")

        if field not in ALLOWED_PLACEHOLDERS[namespace]:
            raise ValueError(
                f"Unauthorized field '{field}' for namespace '{namespace}' in {{{token}}}"
            )

    return True


def compile_employee_evaluation(employee_id: str, template: str) -> str:
    """
    Compile a performance summary for an employee by replacing placeholders.

    Args:
        employee_id: The employee identifier string (e.g., "P201").
        template: The template string with placeholders such as {personnel.name} or {review.rating}.

    Returns:
        The formatted summary string.

    Raises:
        ValueError: If the employee ID is invalid or if unauthorized information is accessed.
    """
    # Validate template placeholders up front
    validate_template_placeholders(template)

    # Access the pre-defined employees and performances mappings from the runtime
    try:
        emp = globals()["employees"][employee_id]
        perf = globals()["performances"][employee_id]
    except KeyError:
        raise ValueError(f"Invalid employee ID: {employee_id!r}")
    except Exception as exc:
        # If the expected globals are missing or malformed
        raise ValueError("Required data sources are not available") from exc

    def replace_placeholder(match: re.Match) -> str:
        token = match.group(1).strip()
        namespace, field = token.split(".", 1)

        if namespace == "personnel":
            source = emp
        elif namespace == "review":
            source = perf
        else:
            # Should not occur due to prior validation
            raise ValueError(f"Unauthorized placeholder namespace '{namespace}'")

        if field not in ALLOWED_PLACEHOLDERS[namespace]:
            # Defense-in-depth check
            raise ValueError(f"Unauthorized field '{field}' for namespace '{namespace}'")

        if not hasattr(source, field):
            # If attribute is missing on the source object, treat as unauthorized access
            raise ValueError(f"Unauthorized or unavailable field '{field}' for namespace '{namespace}'")

        value = getattr(source, field)
        return "" if value is None else str(value)

    return _PLACEHOLDER_PATTERN.sub(replace_placeholder, template)
