import logging
import re

logger = logging.getLogger(__name__)


def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generate a human-readable performance summary for an employee by interpolating
    placeholders in the provided template string.

    The function expects the existence of global mappings `employees` and `performances`,
    where:
      - employees: Dict[str, Employee]
      - performances: Dict[str, Performance], keyed by employee ID

    Supported placeholders:
      - {employee.name}
      - {employee.position}
      - {employee.department}
      - {employee.emp_id}
      - {performance.rating}

    Forbidden (sensitive) placeholders:
      - {performance.feedback}
      - {performance.bonus}

    Placeholder syntax:
      - Each placeholder must be enclosed in curly braces and consist of two parts
        separated by a dot: "<object>.<attribute>".
      - Supported objects are "employee" and "performance".
      - Using an unsupported object or attribute raises ValueError.

    Error handling:
      - Raises ValueError when:
          * arguments are not strings,
          * required globals are missing,
          * the employee ID has no performance entry,
          * an unsupported/unknown placeholder is used,
          * sensitive information is requested,
          * malformed/unmatched braces are present,
          * or any other error occurs during processing.
      - All exceptions are logged via the module logger before raising ValueError.

    Example:
      >>> template = "{employee.name} ({employee.position}) has a rating of {performance.rating}."
      >>> generate_employee_summary("E101", template)
      'John Doe (Senior Software Engineer) has a rating of 4.3.'

    Args:
        employee_id: The ID of the employee whose summary is requested.
        template: The summary template containing placeholders to be replaced.

    Returns:
        The generated summary string with placeholders replaced.

    Raises:
        ValueError: If validation fails, data is missing, placeholders are invalid,
                    or sensitive information is accessed.
    """
    try:
        # --- Guard clauses: type and basic value validation ---
        if not isinstance(employee_id, str):
            raise ValueError("employee_id must be a string.")
        if not isinstance(template, str):
            raise ValueError("template must be a string.")

        emp_id_normalized = employee_id.strip()
        if not emp_id_normalized:
            raise ValueError("employee_id cannot be empty or whitespace.")

        # --- Guard clauses: required globals exist ---
        if "performances" not in globals() or "employees" not in globals():
            raise ValueError("Required data (employees, performances) not available.")

        # Retrieve global data safely.
        performances_map = globals().get("performances")
        employees_map = globals().get("employees")

        # --- Guard clauses: mapping presence and employee existence ---
        if performances_map is None or employees_map is None:
            raise ValueError("Employee or performance data sources are unavailable.")
        if emp_id_normalized not in performances_map:
            raise ValueError(f"Performance data not found for employee ID: {emp_id_normalized}")
        if emp_id_normalized not in employees_map:
            raise ValueError(f"Employee data not found for employee ID: {emp_id_normalized}")

        # Obtain the domain objects for substitution.
        performance_obj = performances_map[emp_id_normalized]
        employee_obj = performance_obj.employee

        # Allowed and sensitive attributes configuration.
        allowed_employee_fields = {"name", "position", "department", "emp_id"}
        allowed_performance_fields = {"rating"}
        sensitive_fields = {"feedback", "bonus"}

        # Placeholder matching pattern: captures content inside single level braces.
        placeholder_pattern = re.compile(r"\{([^{}]+)\}")

        # --- Guard clauses: validate placeholders before substitution ---
        placeholders = [m.group(1).strip() for m in placeholder_pattern.finditer(template)]

        for key in placeholders:
            if "." not in key:
                raise ValueError(f"Unsupported placeholder: {key}")

            obj, attr = (part.strip() for part in key.split(".", 1))

            if attr in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed.")

            if obj == "employee":
                if attr not in allowed_employee_fields:
                    raise ValueError(f"Unsupported employee attribute: {attr}")
            elif obj == "performance":
                if attr not in allowed_performance_fields:
                    raise ValueError(f"Unsupported performance attribute: {attr}")
            else:
                raise ValueError(f"Unsupported placeholder object: {obj}")

        # Detect any unmatched or malformed braces not captured as valid placeholders.
        template_without_valid = placeholder_pattern.sub("", template)
        if "{" in template_without_valid or "}" in template_without_valid:
            raise ValueError("Template contains malformed or unmatched braces.")

        # --- Substitution pass ---
        def replace_placeholder_match(match: re.Match) -> str:
            """Replace a single placeholder occurrence with its corresponding value."""
            placeholder_key = match.group(1).strip()
            obj, attr = (part.strip() for part in placeholder_key.split(".", 1))

            if obj == "employee":
                value = getattr(employee_obj, attr)
            else:  # obj == "performance" (validated above)
                value = getattr(performance_obj, attr)

            return str(value)

        rendered_summary = placeholder_pattern.sub(replace_placeholder_match, template)

        return rendered_summary

    except Exception as exc:
        # Safely log the exception without exposing sensitive data.
        logger.error(
            "Error generating employee summary (employee_id=%s): %s: %s",
            employee_id,
            exc.__class__.__name__,
            str(exc),
            exc_info=True,
        )
        # Convert all errors to ValueError as required by the specification.
        raise ValueError(str(exc))
