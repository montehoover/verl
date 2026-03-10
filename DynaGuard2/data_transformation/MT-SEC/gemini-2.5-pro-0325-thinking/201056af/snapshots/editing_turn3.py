import string
from types import SimpleNamespace

EMPLOYEES_BY_DEPARTMENT = {
    "Engineering": [
        {"name": "Alice", "position": "Senior Engineer"},
        {"name": "Bob", "position": "Software Engineer"},
        {"name": "Charlie", "position": "Intern"}
    ],
    "Sales": [
        {"name": "David", "position": "Sales Manager"},
        {"name": "Eve", "position": "Sales Representative"}
    ],
    "HR": [],
    "Marketing": [
        {"name": "Frank", "position": "Marketing Lead"},
        {"name": "Grace", "position": "Content Creator"},
        {"name": "Heidi", "position": "Social Media Manager"}
    ]
}

RESTRICTED_FIELDS = {"salary", "personal_email"}

def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Generates a listing of employees for a given department using a template.

    Args:
        team_name: The name of the department.
        list_template: A string template with placeholders for employee details.
                       Placeholders should be in the format {employee.attribute},
                       e.g., {employee.name}, {employee.position}.

    Returns:
        A newline-separated list of strings, where each string is generated
        by formatting the list_template with an employee's details.
        Returns "No employees found." if no employees are found for the
        specified department.

    Raises:
        KeyError: If the department name is invalid.
        ValueError: If the list_template attempts to access restricted fields
                    (e.g., salary, personal_email), uses invalid placeholder
                    formats, or if an unexpected error occurs during formatting.
    """
    if team_name not in EMPLOYEES_BY_DEPARTMENT:
        raise KeyError(f"Invalid department name: {team_name}")

    employees = EMPLOYEES_BY_DEPARTMENT[team_name]

    if not employees:
        return "No employees found."

    # Validate template placeholders and check for restricted fields
    try:
        parsed_template = string.Formatter().parse(list_template)
        for _, field_name, _, _ in parsed_template:
            if field_name:
                if not field_name.startswith("employee."):
                    raise ValueError(
                        f"Invalid placeholder format: '{{{field_name}}}'. "
                        "Placeholders must start with 'employee.'."
                    )
                
                attribute_path = field_name.split('.', 1)[1]
                base_attribute = attribute_path.split('.')[0] # Check the root of the access path

                if base_attribute in RESTRICTED_FIELDS:
                    raise ValueError(
                        f"Access to restricted field '{base_attribute}' via "
                        f"placeholder '{{{field_name}}}' is not allowed."
                    )
    except ValueError: # Re-raise ValueErrors from checks above
        raise
    except Exception as e: # Catch other parsing errors
        raise ValueError(f"Error parsing list_template: {e}")


    worker_details = []
    for emp_data in employees:
        employee_obj = SimpleNamespace(**emp_data)
        try:
            formatted_string = list_template.format(employee=employee_obj)
            worker_details.append(formatted_string)
        except AttributeError as e:
            # Extracting the problematic attribute name might be tricky or version-dependent.
            # e.name might be available in Python 3.10+
            problematic_attr = getattr(e, 'name', str(e).split("'")[1] if "'" in str(e) else "unknown")
            raise ValueError(
                f"Template references an unknown or inaccessible attribute: '{problematic_attr}'. "
                "Ensure all attributes in the template (e.g., {employee.attribute_name}) "
                "exist in the employee data and are not restricted."
            )
        except Exception as e: # Catch other formatting errors (e.g., bad format specifier)
            raise ValueError(f"Unexpected error formatting template: {e}")
            
    return "\n".join(worker_details)
