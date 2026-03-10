from string import Formatter

def create_employee_directory(department: str, template: str) -> str:
    """
    Generate a directory string for employees in the specified department,
    replacing placeholders in the template with employee data.

    Allowed placeholders:
      - {employee.name}
      - {employee.position}
      - {employee.department}

    Sensitive fields (access will raise ValueError):
      - {employee.salary}
      - {employee.personal_email}

    Args:
        department: The department name to filter employees by.
        template: The template string containing placeholders.

    Returns:
        A newline-separated string with one line per matching employee.

    Raises:
        ValueError: If inputs are invalid, if the template attempts to access
        sensitive or unknown fields, or on any other error.
    """
    try:
        if not isinstance(department, str) or not isinstance(template, str):
            raise ValueError("department and template must be strings")

        allowed_attrs = {"name", "position", "department"}
        sensitive_attrs = {"salary", "personal_email"}

        # Validate template fields
        formatter = Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is None:
                continue  # literal text
            if field_name == "":
                raise ValueError("Empty placeholder is not allowed")

            # Only allow placeholders of the exact form: employee.<attr>
            if not field_name.startswith("employee."):
                raise ValueError("Invalid placeholder; only {employee.<field>} is allowed")

            parts = field_name.split(".")
            if len(parts) != 2 or not parts[1]:
                raise ValueError("Invalid placeholder format; use {employee.<field>}")

            attr = parts[1]

            # Disallow additional indexing or attribute chaining
            if any(ch in attr for ch in "[]()."):
                raise ValueError("Chained or indexed access is not allowed")

            if attr in sensitive_attrs:
                raise ValueError("Attempt to access sensitive information")

            if attr not in allowed_attrs:
                raise ValueError(f"Unknown or disallowed field: {attr}")

        # Build the directory for the given department
        try:
            data = globals().get("employees", None)
            if not isinstance(data, (list, tuple)):
                raise ValueError("Employee data is unavailable or invalid")
            matched_employees = [emp for emp in data if getattr(emp, "department", None) == department]
        except ValueError:
            # Re-raise our explicit ValueError
            raise
        except Exception as e:
            # In case global 'employees' is missing or malformed
            raise ValueError("Employee data is unavailable or invalid") from e

        lines = []
        for emp in matched_employees:
            # Safe because we've validated the placeholders to be limited to allowed attributes
            try:
                lines.append(template.format(employee=emp))
            except Exception as e:
                raise ValueError("Failed to render template") from e

        return "\n".join(lines)

    except ValueError:
        # Re-raise explicit ValueError as-is
        raise
    except Exception as e:
        # Wrap any unexpected errors
        raise ValueError(str(e))
