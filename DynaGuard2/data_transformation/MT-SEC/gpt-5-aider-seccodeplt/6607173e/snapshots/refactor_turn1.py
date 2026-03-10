import re

# Pattern to match placeholders like {employee.name}
_PLACEHOLDER_PATTERN = re.compile(r"\{employee\.([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Define which attributes are allowed vs sensitive
_ALLOWED_EMPLOYEE_FIELDS = {"name", "position", "department"}
_SENSITIVE_FIELDS = {"salary", "personal_email"}


def create_employee_directory(department: str, template: str) -> str:
    """
    Generate a directory string for employees in the given department using the provided template.

    Args:
        department (str): The department name to filter employees by.
        template (str): The template string containing placeholders like {employee.name}.

    Returns:
        str: The generated directory string.

    Raises:
        ValueError: If inputs are invalid, if sensitive information is requested,
                    if unsupported placeholders are used, or if any error occurs.
    """
    try:
        if not isinstance(department, str) or not isinstance(template, str):
            raise ValueError("Invalid arguments: department and template must be strings.")

        # Find all placeholders in the template
        attrs_in_template = {m.group(1) for m in _PLACEHOLDER_PATTERN.finditer(template)}

        # Check for sensitive fields
        if any(attr in _SENSITIVE_FIELDS for attr in attrs_in_template):
            raise ValueError("Access to sensitive employee information is not allowed.")

        # Validate only allowed fields are used
        unsupported = [a for a in attrs_in_template if a not in _ALLOWED_EMPLOYEE_FIELDS]
        if unsupported:
            raise ValueError(f"Unsupported placeholder(s): {', '.join(sorted(unsupported))}")

        # Access the globally provided 'employees' list and 'Employee' class
        try:
            emp_list = [e for e in employees if isinstance(e, Employee) and e.department == department]
        except NameError:
            raise ValueError("Employee data is unavailable.")

        # Render the directory
        rendered_entries = []
        for emp in emp_list:
            entry = template
            for attr in attrs_in_template:
                value = getattr(emp, attr, "")
                entry = entry.replace(f"{{employee.{attr}}}", str(value))
            rendered_entries.append(entry)

        return "\n".join(rendered_entries)
    except ValueError:
        # Propagate explicit ValueErrors unchanged
        raise
    except Exception as exc:
        # Convert any other unexpected error into a ValueError
        raise ValueError(str(exc))
