from typing import Any, Mapping, Optional, Union


def _format_salary(value: Any) -> str:
    """
    Formats a salary value. Accepts numbers or strings (e.g., "$70,000", "70000").
    Returns a human-friendly string with a dollar sign and thousand separators.
    """
    if value is None:
        return "N/A"

    # If it's already numeric
    if isinstance(value, (int, float)):
        return f"${value:,.2f}"

    # Try to parse if it's a string
    if isinstance(value, str):
        stripped = value.strip().replace("$", "").replace(",", "")
        try:
            number = float(stripped)
            return f"${number:,.2f}"
        except ValueError:
            # Fallback to raw string if unparsable
            return value

    # Fallback for other types
    return str(value)


def _canonicalize_ssn(ssn: Any) -> Optional[str]:
    """
    Canonicalizes SSN-like input to the standard ###-##-#### format if possible.
    Returns None if no usable SSN digits found.
    """
    if ssn is None:
        return None

    digits = [c for c in str(ssn) if c.isdigit()]
    if not digits:
        return None

    if len(digits) == 9:
        return f"{digits[0]}{digits[1]}{digits[2]}-{digits[3]}{digits[4]}-{digits[5]}{digits[6]}{digits[7]}{digits[8]}"
    # If not exactly 9 digits, return the original string representation
    return str(ssn)


def display_employee_info(
    staff: Mapping[str, Any],
    include_sensitive: bool = False,
) -> str:
    """
    Returns a formatted string of employee details.

    Parameters:
    - staff: Mapping with keys like "name", "position", "salary", "department", "social_security_number".
    - include_sensitive: When True, includes sensitive fields (e.g., social_security_number) in the output.
                         When False, sensitive fields are excluded from the output.

    Example:
        display_employee_info(
            {
                "name": "Jane Doe",
                "position": "Engineer",
                "salary": 95000,
                "department": "R&D",
                "social_security_number": "123-45-6789",
            },
            include_sensitive=False
        )
    """
    if not isinstance(staff, Mapping):
        raise TypeError("staff must be a mapping/dictionary")

    name = staff.get("name", "N/A")
    position = staff.get("position", "N/A")
    salary = _format_salary(staff.get("salary"))
    department = staff.get("department", "N/A")

    lines = [
        f"Name: {name}",
        f"Position: {position}",
        f"Department: {department}",
        f"Salary: {salary}",
    ]

    if include_sensitive:
        ssn = _canonicalize_ssn(staff.get("social_security_number"))
        lines.append(f"Social Security Number: {ssn if ssn is not None else 'N/A'}")

    return "\n".join(lines)
