import re

class Employee:
    """Represents an employee with their professional and personal details."""
    def __init__(self, name: str, position: str, department: str, salary: int, personal_email: str):
        """
        Initializes an Employee object.

        Args:
            name: The employee's full name.
            position: The employee's job position.
            department: The department the employee belongs to.
            salary: The employee's annual salary.
            personal_email: The employee's personal email address.
        """
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary  # Sensitive attribute
        self.personal_email = personal_email  # Sensitive attribute


# Global list of employees for the directory system.
employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]


def _filter_employees_by_department(all_employees: list[Employee], department_name: str) -> list[Employee]:
    """
    Filters a list of Employee objects to include only those in the specified department.

    Args:
        all_employees: A list of all Employee objects.
        department_name: The name of the department to filter by.

    Returns:
        A new list containing Employee objects belonging to the specified department.
    """
    return [emp for emp in all_employees if emp.department == department_name]


def _process_employee_template(employee: Employee, template: str, sensitive_fields: list[str]) -> str:
    """
    Processes a template string for a single employee, replacing placeholders
    with employee data while ensuring sensitive information is not exposed.

    Args:
        employee: The Employee object whose data will be used.
        template: The template string containing placeholders (e.g., "{employee.name}").
        sensitive_fields: A list of attribute names considered sensitive and
                          not allowed in the template.

    Returns:
        The processed string with placeholders replaced by employee data.

    Raises:
        ValueError: If the template attempts to access sensitive or non-existent
                    fields, or if any other template processing error occurs.
    """
    
    # 1. Initial security check: Ensure template does not explicitly ask for sensitive fields.
    for field in sensitive_fields:
        if f"{{employee.{field}}}" in template:
            raise ValueError(f"Access to sensitive field '{field}' is not allowed in template.")

    # 2. Define a dictionary of allowed, non-sensitive attributes for template population.
    # This acts as an allow-list for template placeholders.
    allowed_data_for_template = {
        "name": employee.name,
        "position": employee.position,
        "department": employee.department  # Department can be useful for context in the entry.
    }

    # 3. Validate all placeholders found in the template.
    # Placeholders must correspond to keys in `allowed_data_for_template`.
    placeholders = re.findall(r"\{employee\.(\w+)\}", template)
    for placeholder_attribute in placeholders:
        if placeholder_attribute in sensitive_fields:
            # This is a defense-in-depth check, as step 1 should catch direct sensitive field requests.
            # However, this ensures that even if a sensitive field was missed in step 1 (e.g., due to
            # a complex template structure not caught by simple string search), it's caught here.
            raise ValueError(
                f"Access to sensitive field '{placeholder_attribute}' via placeholder is not allowed."
            )
        if placeholder_attribute not in allowed_data_for_template:
            raise ValueError(
                f"Access to unspecified or disallowed field '{placeholder_attribute}' is not allowed."
            )

    # 4. Perform the replacement of placeholders with actual data.
    processed_template = template
    try:
        for attr_key, attr_value in allowed_data_for_template.items():
            # Replace all occurrences of the placeholder for the current attribute.
            processed_template = processed_template.replace(f"{{employee.{attr_key}}}", str(attr_value))
        
        # Sanity check: After all replacements, verify that no placeholders for *allowed*
        # fields remain. This could indicate an issue in the replacement logic or an
        # unexpected template format that wasn't fully processed.
        final_check_placeholders = re.findall(r"\{employee\.(\w+)\}", processed_template)
        problematic_remaining = [
            p for p in final_check_placeholders if p in allowed_data_for_template
        ]
        if problematic_remaining:
             raise ValueError(
                 f"Template processing error: Placeholder for allowed field "
                 f"'{problematic_remaining[0]}' remained after replacement. "
                 f"Ensure template placeholders are correctly formatted."
            )

    except Exception as e:
        # Catch any unexpected errors during string operations.
        raise ValueError(
            f"Error during template string replacement for employee {employee.name}: {str(e)}"
        )

    return processed_template


def create_employee_directory(department: str, template: str) -> str:
    """
    Generates a directory string for all employees in a specified department,
    using a given template for each employee's entry.

    This function orchestrates the filtering of employees by department and
    the processing of the template for each selected employee. It defines
    which employee attributes are considered sensitive and should not be
    accessible via the template.

    Args:
        department: The name of the department for which to generate the directory.
        template: The template string to use for each employee's entry.
                  Placeholders like "{employee.name}" or "{employee.position}"
                  will be replaced with the employee's actual data.

    Returns:
        A string containing all directory entries, separated by newlines.
        Returns an empty string if no employees are found in the department.

    Raises:
        ValueError: If the template attempts to access sensitive information
                    (e.g., salary, personal_email), or if any other error occurs
                    during template processing or employee data handling.
    """
    directory_entries = []
    
    # Define sensitive fields at this level, as it's a policy of the directory creation.
    sensitive_fields = ["salary", "personal_email"]

    # Filter employees belonging to the specified department.
    department_employees = _filter_employees_by_department(employees, department)

    if not department_employees:
        # If no employees are found in the department, return an empty string.
        return ""

    # Process the template for each employee in the filtered list.
    for emp in department_employees:
        try:
            # Generate the directory entry for the current employee.
            # The _process_employee_template function handles validation and replacement.
            processed_entry = _process_employee_template(emp, template, sensitive_fields)
            directory_entries.append(processed_entry)
        except ValueError:
            # If _process_employee_template raises a ValueError (e.g., due to
            # sensitive data access or malformed template), propagate the error.
            raise
        # No other specific exceptions are expected here, as _process_employee_template
        # should convert its internal errors to ValueError.

    # Join all individual directory entries into a single string, separated by newlines.
    return "\n".join(directory_entries)
