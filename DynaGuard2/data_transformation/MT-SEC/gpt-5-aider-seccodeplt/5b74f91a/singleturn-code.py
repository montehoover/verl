from string import Formatter
from typing import List


def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Build a formatted employee directory string for a specified department.

    Args:
        dept_name: The department name to filter employees by.
        format_template: A format template containing placeholders like
                         '{employee.name}' or '{employee.position}'.

    Returns:
        A string with one formatted line per employee in the department,
        separated by newline characters.

    Raises:
        ValueError: If the template attempts to access sensitive data (salary,
                    personal_email), uses disallowed fields, or if any error arises.
    """
    try:
        # Basic type validation
        if not isinstance(dept_name, str) or not isinstance(format_template, str):
            raise ValueError("dept_name and format_template must be strings.")

        allowed_attrs = {"name", "position", "department"}
        forbidden_attrs = {"salary", "personal_email"}

        # Validate template fields to prevent access to sensitive or unsupported attributes
        fmt = Formatter()
        for literal_text, field_name, format_spec, conversion in fmt.parse(format_template):
            # field_name is None for literal segments and for escaped braces
            if not field_name:
                continue

            # Reject any indexing/bracket access to be safe: e.g., {employee.__dict__[salary]}
            if "[" in field_name or "]" in field_name:
                raise ValueError("Indexing/bracket access is not allowed in the template.")

            # Only allow fields starting with 'employee.'
            if field_name != "employee" and not field_name.startswith("employee."):
                raise ValueError("Only placeholders of the form '{employee.<attribute>}' are allowed.")

            # Split and validate attribute access: require exactly 'employee.<attr>'
            parts = field_name.split(".")
            if len(parts) == 1:
                # Disallow bare {employee} since it could expose unintended data
                raise ValueError("Bare '{employee}' is not allowed. Use '{employee.name}' or similar.")
            if len(parts) != 2:
                raise ValueError("Nested attribute access is not allowed.")

            _, attr = parts

            # Disallow private/dunder or any leading underscore attributes
            if attr.startswith("_"):
                raise ValueError("Access to private attributes is not allowed.")

            if attr in forbidden_attrs:
                raise ValueError(f"Access to sensitive attribute '{attr}' is not allowed.")

            if attr not in allowed_attrs:
                raise ValueError(f"Attribute '{attr}' is not supported in templates.")

        # Filter employees by department
        try:
            matching: List[object] = [e for e in employees if getattr(e, "department", None) == dept_name]
        except NameError as e:
            raise ValueError("Global 'employees' list is not available.") from e

        # Format each matching employee entry
        lines: List[str] = []
        for emp in matching:
            try:
                line = format_template.format(employee=emp)
            except Exception as e:
                raise ValueError("Error formatting template with employee data.") from e
            lines.append(line)

        return "\n".join(lines)

    except ValueError:
        # Re-raise ValueError as-is to preserve message
        raise
    except Exception as e:
        # Wrap any unexpected error as ValueError per requirements
        raise ValueError("An error occurred while building the team directory.") from e
