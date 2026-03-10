def print_employee_details(worker_id, format_template):
    """
    Return a formatted employee report string based on the provided template.

    The template can include placeholders:
      - {worker_id}
      - {worker_name}
      - {position}
      - {department}

    Unknown placeholders are left unchanged. Accessing unauthorized placeholders
    like {feedback} or {bonus} will raise a ValueError.

    Raises:
        ValueError: If worker_id is invalid or not found, or if unauthorized
                    placeholders are referenced.
    """
    # Validate input worker_id and employees registry
    if not isinstance(worker_id, str):
        raise ValueError("Invalid worker ID")

    wid = worker_id.strip()
    if not wid or "employees" not in globals() or not isinstance(globals()["employees"], dict):
        raise ValueError("Invalid worker ID")

    registry = globals()["employees"]

    if wid not in registry:
        raise ValueError("Invalid worker ID")

    emp = registry[wid]

    # Prepare allowed placeholder values
    placeholders = {
        "worker_id": emp.emp_id,
        "worker_name": emp.name,
        "position": emp.position,
        "department": emp.department,
    }

    # Ensure template is a string
    template = str(format_template)

    # Check for unauthorized placeholders (feedback, bonus)
    try:
        from string import Formatter
        formatter = Formatter()
        unauthorized = {"feedback", "bonus"}
        for _, field_name, _, _ in formatter.parse(template):
            if field_name:
                # Only consider the base name before attribute/index access
                base_name = field_name.split(".", 1)[0].split("[", 1)[0]
                if base_name in unauthorized:
                    raise ValueError("Unauthorized placeholder accessed")
    except ValueError:
        # Re-raise ValueErrors from unauthorized access as-is
        raise

    # Safe formatting: leave unknown placeholders unchanged
    class _SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return template.format_map(_SafeDict(placeholders))
