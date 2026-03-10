from typing import Any, Dict


def get_employee_details(employee: Dict[str, Any], exclude_sensitive: bool = True) -> str:
    """
    Return a formatted string of employee details.

    Parameters:
        employee (dict): Dictionary with keys like:
            - name
            - position
            - salary
            - department
            - social_security_number
        exclude_sensitive (bool): If True, omit sensitive fields (e.g., social_security_number)
                                  from the output. Defaults to True.

    Returns:
        str: Formatted employee details.
    """
    def _format(value: Any, default: str = "N/A") -> str:
        if value is None:
            return default
        return str(value)

    name = _format(employee.get("name"))
    position = _format(employee.get("position"))
    salary = _format(employee.get("salary"))
    department = _format(employee.get("department"))
    ssn = _format(employee.get("social_security_number"))

    lines = [
        f"Name: {name}",
        f"Position: {position}",
        f"Department: {department}",
        f"Salary: {salary}",
    ]

    if not exclude_sensitive and employee.get("social_security_number") is not None:
        lines.append(f"Social Security Number: {ssn}")

    return "\n".join(lines)
