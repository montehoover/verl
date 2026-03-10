from dataclasses import dataclass, fields
from typing import Dict, Any
import string
import re


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    title: str
    department: str
    email: str | None = None


# Predefined dictionary of employees keyed by string worker ID
EMPLOYEES: Dict[str, Employee] = {
    "1001": Employee(id="1001", name="Alice Johnson", title="Software Engineer", department="Engineering", email="alice.johnson@example.com"),
    "1002": Employee(id="1002", name="Bob Smith", title="Product Manager", department="Product", email="bob.smith@example.com"),
    "1003": Employee(id="1003", name="Carol Lee", title="Data Analyst", department="Data", email="carol.lee@example.com"),
}


def get_employee_by_id(worker_id: Any) -> Employee:
    """
    Retrieve an Employee by their worker ID.

    Args:
        worker_id: The identifier of the employee (int, str, or any type convertible to str).

    Returns:
        The corresponding Employee object.

    Raises:
        ValueError: If no employee exists with the given worker ID.
    """
    key = str(worker_id)
    try:
        return EMPLOYEES[key]
    except KeyError as exc:
        raise ValueError(f"Employee with ID {worker_id} does not exist.") from exc


def validate_report_template(template: str) -> bool:
    """
    Validate that all placeholders in the provided template are authorized and correctly formed.

    Allowed placeholders:
      - {worker.<field>} where <field> is one of Employee's dataclass fields.
      - {assessment.rating}, {assessment.feedback}, {assessment.bonus}

    Escaped braces {{ and }} are supported and ignored by validation.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an unauthorized or incorrectly formatted placeholder is found.
    """
    formatter = string.Formatter()
    worker_fields = {f.name for f in fields(Employee)}
    allowed_assessment_fields = {"rating", "feedback", "bonus"}
    pattern = re.compile(r"^(?P<ns>worker|assessment)\.(?P<attr>[A-Za-z_][A-Za-z0-9_]*)$")

    for _, field_name, format_spec, conversion in formatter.parse(template):
        # literal segments yield field_name=None
        if field_name is None:
            continue

        # Disallow format conversions or specs in placeholders (e.g., !r, :^10)
        if conversion not in (None, ""):
            raise ValueError(f"Unsupported conversion '!{conversion}' in placeholder {{{field_name}}}.")
        if format_spec:
            raise ValueError(f"Unsupported format spec ': {format_spec}' in placeholder {{{field_name}}}.")

        match = pattern.match(field_name)
        if not match:
            raise ValueError(f"Invalid placeholder '{{{field_name}}}'. Use dotted paths like '{{worker.name}}' or '{{assessment.rating}}'.")

        ns = match.group("ns")
        attr = match.group("attr")

        if ns == "worker":
            if attr not in worker_fields:
                raise ValueError(f"Unauthorized worker placeholder '{{worker.{attr}}}'. Allowed: {sorted(worker_fields)}")
        elif ns == "assessment":
            if attr not in allowed_assessment_fields:
                allowed = sorted(allowed_assessment_fields)
                raise ValueError(f"Unauthorized assessment placeholder '{{assessment.{attr}}}'. Allowed: {allowed}")
        else:
            # Should not occur due to regex, but keep for safety
            raise ValueError(f"Unauthorized placeholder namespace '{ns}' in '{{{field_name}}}'.")

    return True


@dataclass(frozen=True)
class Performance:
    worker: Employee
    rating: float
    feedback: str
    bonus: float


# Predefined performance data keyed by worker ID corresponding to EMPLOYEES
PERFORMANCES: Dict[str, Performance] = {
    "1001": Performance(worker=EMPLOYEES["1001"], rating=4.8, feedback="Led successful feature releases", bonus=3000),
    "1002": Performance(worker=EMPLOYEES["1002"], rating=4.2, feedback="Coordinated cross-team roadmaps", bonus=2800),
    "1003": Performance(worker=EMPLOYEES["1003"], rating=4.0, feedback="Delivered insightful analyses", bonus=2500),
}


def create_employee_report(worker_id: str, template: str) -> str:
    """
    Generate a formatted performance summary for a worker.

    Args:
        worker_id: The string identifier of the worker.
        template: A template string containing placeholders like {worker.name} or {assessment.rating}.

    Returns:
        A formatted summary string with placeholders replaced.

    Raises:
        ValueError: If the worker ID is invalid, template contains unauthorized placeholders,
                    or performance data is required by the template but missing.
    """
    # Validate the template placeholders for authorization and shape
    validate_report_template(template)

    # Retrieve the employee or raise
    employee = get_employee_by_id(worker_id)

    # Determine if assessment placeholders are present
    formatter = string.Formatter()
    needs_assessment = any(
        (field_name is not None) and field_name.startswith("assessment.")
        for _, field_name, _, _ in formatter.parse(template)
    )

    assessment = PERFORMANCES.get(worker_id)
    if needs_assessment and assessment is None:
        raise ValueError(f"No performance data found for worker ID {worker_id}.")

    # Perform formatting with controlled objects only
    return template.format(worker=employee, assessment=assessment)
