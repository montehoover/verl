from typing import Any, Dict


def get_employee_details(emp: Dict[str, Any], exclude_sensitive: bool = True) -> str:
    """
    Return a formatted string of employee details.

    Parameters:
        emp (dict): A dictionary with possible keys:
            - name
            - position
            - salary
            - department
            - social_security_number
        exclude_sensitive (bool): If True, exclude sensitive fields (e.g., social_security_number)
            from the output. Defaults to True.

    Returns:
        str: A multi-line string with formatted employee details.
    """
    fields_order = [
        ("Name", "name"),
        ("Position", "position"),
        ("Department", "department"),
        ("Salary", "salary"),
        ("Social Security Number", "social_security_number"),
    ]

    sensitive_keys = {"social_security_number"}

    lines = []
    for label, key in fields_order:
        if exclude_sensitive and key in sensitive_keys:
            continue

        value = emp.get(key, "N/A")

        # Convert non-string values to string safely
        if value is None:
            value_str = "N/A"
        else:
            value_str = str(value)

        lines.append(f"{label}: {value_str}")

    return "\n".join(lines)
