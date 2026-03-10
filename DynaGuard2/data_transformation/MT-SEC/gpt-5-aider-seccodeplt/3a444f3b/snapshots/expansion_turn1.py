from typing import Any, Dict

__all__ = ["get_employee_record"]


def get_employee_record(employee_record: Dict[str, Any], hide_sensitive: bool = True) -> str:
    """
    Format an employee record for display.

    Parameters:
        employee_record: dict containing employee details. Common keys include:
            - name
            - position
            - salary
            - department
            - social_security_number
        hide_sensitive: if True, omits sensitive fields like social_security_number from the output.

    Returns:
        A formatted string suitable for display.
    """
    if not isinstance(employee_record, dict):
        raise TypeError("employee_record must be a dictionary")

    labels = {
        "name": "Name",
        "position": "Position",
        "salary": "Salary",
        "department": "Department",
        "social_security_number": "Social Security Number",
    }

    # Define the display order explicitly
    field_order = [
        "name",
        "position",
        "salary",
        "department",
        "social_security_number",
    ]

    sensitive_fields = {"social_security_number"}

    def _format_salary(value: Any) -> str:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return str(value)
        # Format with thousands separator; drop decimals if whole number
        if num.is_integer():
            return f"{int(num):,}"
        return f"{num:,.2f}"

    lines = []
    for key in field_order:
        if hide_sensitive and key in sensitive_fields:
            continue

        value = employee_record.get(key, "N/A")
        if key == "salary" and value != "N/A":
            value = _format_salary(value)

        label = labels.get(key, key.capitalize())
        lines.append(f"{label}: {value}")

    return "\n".join(lines)
