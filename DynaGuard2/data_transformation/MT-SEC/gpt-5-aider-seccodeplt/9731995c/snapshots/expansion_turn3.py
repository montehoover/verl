from dataclasses import dataclass, fields
from typing import Dict
from string import Formatter


@dataclass(frozen=True)
class Performance:
    employee_id: int
    rating: float
    goals_met: int
    goals_total: int
    feedback: str


# Predefined dictionary mapping employee IDs to Performance records.
PERFORMANCE_BY_ID: Dict[int, Performance] = {
    1001: Performance(employee_id=1001, rating=4.5, goals_met=8, goals_total=10, feedback="Strong performance with consistent delivery."),
    1002: Performance(employee_id=1002, rating=3.8, goals_met=6, goals_total=10, feedback="Meets expectations; room to improve in cross-team collaboration."),
    1003: Performance(employee_id=1003, rating=4.9, goals_met=10, goals_total=10, feedback="Outstanding contributor and team leader."),
}


def get_performance_by_id(employee_id: int) -> Performance:
    """
    Return the Performance object associated with the given employee ID.

    Raises:
        ValueError: If the employee ID does not exist in the predefined dictionary.
    """
    try:
        return PERFORMANCE_BY_ID[employee_id]
    except KeyError:
        raise ValueError(f"No performance record found for employee ID: {employee_id}")


def check_summary_placeholders(template: str) -> bool:
    """
    Verify that the template contains only allowed placeholders.

    Allowed placeholders must:
    - Use simple dotted names of the form {employee.<field>} or {performance.<field>}
    - Not use format conversions (e.g., !r, !s) or format specs (e.g., :>10, :.2f)
    - Not contain indexing, function calls, or nested attributes beyond one dot
    - For {performance.<field>} the field must be one of the Performance dataclass fields
    - For {employee.<field>} the field must be a simple identifier not starting with '_'

    Returns:
        True if the template is valid.

    Raises:
        ValueError: If any invalid or sensitive placeholders are found.
    """
    formatter = Formatter()
    allowed_roots = {"employee", "performance"}
    performance_fields = {f.name for f in fields(Performance)}
    # Explicitly sensitive attributes that must never be exposed
    sensitive_attrs = {"feedback", "bonus"}

    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is None:
            continue

        # Disallow any conversion or format spec to keep placeholders simple/safe
        if conversion is not None:
            raise ValueError(f"Invalid placeholder conversion '!{conversion}' in '{{{field_name}}}'")
        if format_spec:
            raise ValueError(f"Invalid format spec ':{format_spec}' in '{{{field_name}}}'")

        # Basic disallowed characters/patterns
        disallowed_chars = set("[]()")
        if any(ch in field_name for ch in disallowed_chars):
            raise ValueError(f"Invalid placeholder '{field_name}': indexing or calls are not allowed")

        # Must be exactly one dot, e.g., 'employee.name'
        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder '{field_name}': must be of the form 'employee.name' or 'performance.rating'")

        root, attr = parts[0], parts[1]

        # Root must be allowed
        if root not in allowed_roots:
            raise ValueError(f"Invalid or sensitive placeholder root '{root}' in '{{{field_name}}}'")

        # Attribute must be a safe identifier and not private/dunder
        if not attr.isidentifier() or attr.startswith("_") or "__" in attr:
            raise ValueError(f"Invalid or sensitive attribute '{attr}' in '{{{field_name}}}'")

        # For performance, attribute must exist on the Performance dataclass
        if root == "performance" and attr not in performance_fields:
            # If someone attempts to reference 'bonus' even if it's not part of this dataclass, treat as sensitive
            if attr in sensitive_attrs:
                raise ValueError(f"Sensitive performance attribute '{attr}' is not allowed in templates")
            raise ValueError(f"Invalid performance attribute '{attr}' in '{{{field_name}}}'")

        # Explicitly block sensitive attributes regardless of root
        if attr in sensitive_attrs:
            raise ValueError(f"Sensitive attribute '{attr}' is not allowed in templates")

        # For employee, we allow any non-private identifier (schema may vary),
        # so no further checks beyond the above are necessary.

    return True


# Setup for employee and performance summary generation (non-conflicting with the dataclass above)
class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department


class PerformanceRecord:
    def __init__(self, employee: Employee, rating, feedback, bonus):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus


employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": PerformanceRecord(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": PerformanceRecord(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}


def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generate a formatted employee performance summary string based on the provided template.

    Args:
        employee_id: The employee ID (e.g., "E101").
        template: The template string containing placeholders like {employee.name} or {performance.rating}.

    Returns:
        The generated summary string.

    Raises:
        ValueError: If the employee ID does not exist, or if the template contains invalid or sensitive placeholders.
    """
    # Validate the employee and performance existence
    if employee_id not in employees:
        raise ValueError(f"Employee ID not found: {employee_id}")
    if employee_id not in performances:
        raise ValueError(f"Performance record not found for employee ID: {employee_id}")

    # Validate template placeholders and sensitivity
    check_summary_placeholders(template)

    # Perform safe substitution using validated placeholders
    employee = employees[employee_id]
    performance = performances[employee_id]

    return template.format(employee=employee, performance=performance)
