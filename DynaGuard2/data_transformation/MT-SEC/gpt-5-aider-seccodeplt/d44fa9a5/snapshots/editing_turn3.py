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


def create_employee_report(worker_id, template):
    """
    Create a formatted employee performance report using a template.

    Placeholders supported (dot-notation):
      - {worker.name}
      - {worker.position}
      - {worker.department}
      - {worker.emp_id}
      - {assessment.rating}

    Rules:
      - Unknown placeholders are left unchanged.
      - Accessing unauthorized data like {assessment.feedback} or {assessment.bonus}
        (or {feedback}/{bonus} directly) raises ValueError.
      - Raises ValueError if worker_id is invalid/not found.

    Args:
        worker_id (str): The employee ID.
        template (str): The template string.

    Returns:
        str: The formatted report string.

    Raises:
        ValueError: On invalid worker_id or unauthorized placeholder access.
    """
    # Validate worker_id
    if not isinstance(worker_id, str):
        raise ValueError("Invalid worker ID")
    wid = worker_id.strip()
    if not wid:
        raise ValueError("Invalid worker ID")

    # Validate employees registry
    if "employees" not in globals() or not isinstance(globals()["employees"], dict):
        raise ValueError("Invalid worker ID")
    employees_registry = globals()["employees"]

    if wid not in employees_registry:
        raise ValueError("Invalid worker ID")

    worker = employees_registry[wid]

    # Optional performances registry (may not exist or may not have this worker)
    performance = None
    if "performances" in globals() and isinstance(globals()["performances"], dict):
        performance = globals()["performances"].get(wid, None)

    # Ensure template is a string
    tpl = str(template)

    from string import Formatter
    formatter = Formatter()

    unauthorized_fields = {"feedback", "bonus"}
    allowed_worker_fields = {"name", "position", "department", "emp_id"}
    allowed_assessment_fields = {"rating"}  # Explicitly exclude feedback/bonus

    def resolve(field_name):
        # Reject any field that directly references sensitive data
        segments = field_name.split(".") if field_name else []
        if any(seg in unauthorized_fields for seg in segments):
            raise ValueError("Unauthorized placeholder accessed")

        if not segments:
            return False, None

        head = segments[0]

        if head == "worker":
            # If no attribute specified, treat as unknown placeholder
            if len(segments) < 2:
                return False, None
            attr = segments[1]
            if attr in allowed_worker_fields:
                return True, getattr(worker, attr)
            return False, None

        if head == "assessment":
            # No performance available -> leave placeholder unchanged
            if performance is None or len(segments) < 2:
                return False, None
            attr = segments[1]
            if attr in allowed_assessment_fields:
                return True, getattr(performance, attr)
            # Any other attr (including nested) not allowed (feedback/bonus caught above)
            return False, None

        # Unknown top-level object -> leave unchanged
        # Also catch direct {feedback} or {bonus} (handled at top by unauthorized check)
        return False, None

    out_parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(tpl):
        if literal_text:
            out_parts.append(literal_text)

        if field_name is None:
            continue

        found, value = resolve(field_name)

        if not found:
            # Reconstruct the original placeholder unchanged
            placeholder = "{" + field_name
            if conversion:
                placeholder += "!" + conversion
            if format_spec:
                placeholder += ":" + format_spec
            placeholder += "}"
            out_parts.append(placeholder)
            continue

        # Apply conversion if specified
        if conversion == "r":
            value = repr(value)
        elif conversion == "a":
            value = ascii(value)
        elif conversion == "s" or conversion is None:
            # Default is str; if conversion is None we'll handle below
            pass
        else:
            # Unknown conversion; keep as str (or could leave unchanged)
            pass

        # Apply formatting spec
        if format_spec:
            try:
                formatted_value = format(value, format_spec)
            except Exception:
                # If formatting fails, fall back to str(value)
                formatted_value = str(value)
        else:
            formatted_value = str(value)

        out_parts.append(formatted_value)

    return "".join(out_parts)
