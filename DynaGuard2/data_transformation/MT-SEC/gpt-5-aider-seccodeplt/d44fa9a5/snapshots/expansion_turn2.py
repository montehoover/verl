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
      - {assessment.rating}

    Escaped braces {{ and }} are supported and ignored by validation.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an unauthorized or incorrectly formatted placeholder is found.
    """
    formatter = string.Formatter()
    worker_fields = {f.name for f in fields(Employee)}
    allowed_assessment_fields = {"rating"}
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
