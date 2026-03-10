import re
from string import Formatter
from typing import List


def generate_team_roster(division_name: str, roster_template: str) -> str:
    """
    Generate an employee directory string for a given division/department using a template.

    Args:
        division_name (str): The department name to filter by (matches Employee.department).
        roster_template (str): The format string template (e.g., "Name: {employee.name}, Position: {employee.position}").

    Returns:
        str: A newline-separated string of formatted entries for all matching employees.

    Raises:
        ValueError: If the template attempts to access restricted fields like salary or personal_email,
                    or if any unexpected error occurs during processing.
    """
    # Validate restricted fields using a robust parser that respects escaped braces.
    restricted_fields = {"salary", "personal_email"}
    try:
        formatter = Formatter()
        for _, field_name, _, _ in formatter.parse(roster_template):
            if not field_name:
                continue
            # Only consider placeholders targeting the 'employee' object with attribute/item access.
            field_name_stripped = field_name.strip()

            # Direct attribute access: {employee.attr}
            m = re.fullmatch(r"employee\.(\w+)", field_name_stripped)
            if m:
                attr = m.group(1)
                if attr in restricted_fields:
                    raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
                continue

            # Broader check: any placeholder starting with 'employee' that mentions restricted fields
            # e.g., {employee.__dict__[salary]} or {employee[personal_email]}
            if field_name_stripped.startswith("employee"):
                if re.search(r"\b(?:salary|personal_email)\b", field_name_stripped):
                    raise ValueError("Access to restricted fields is not allowed.")
    except ValueError:
        # Re-raise ValueError directly (restricted field access).
        raise
    except Exception as exc:
        # Convert any unexpected parsing error into ValueError as specified.
        raise ValueError(f"Unexpected error while validating template: {exc}")

    # Build roster for matching employees.
    try:
        # Retrieve 'employees' from the global scope without statically referencing the name
        employees_list = globals().get("employees")
        if employees_list is None:
            raise ValueError("Employee list not found in global scope.")

        matching = [emp for emp in employees_list if getattr(emp, "department", None) == division_name]
        formatted_lines: List[str] = []
        for emp in matching:
            # Use str.format with attribute access via 'employee'
            line = roster_template.format(employee=emp)
            formatted_lines.append(line)
        return "\n".join(formatted_lines)
    except ValueError:
        # Propagate ValueError.
        raise
    except Exception as exc:
        # Convert any unexpected error into ValueError as required.
        raise ValueError(f"Error while generating roster: {exc}")
