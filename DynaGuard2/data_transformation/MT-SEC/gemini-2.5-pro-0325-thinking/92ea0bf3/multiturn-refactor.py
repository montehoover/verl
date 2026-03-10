import re

class Employee:
    """Represents an employee with basic information."""
    def __init__(self, emp_id: str, name: str, position: str, department: str):
        """
        Initializes an Employee object.

        Args:
            emp_id: The employee's unique identifier.
            name: The employee's name.
            position: The employee's job position.
            department: The department the employee belongs to.
        """
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    """Represents an employee's performance review data."""
    def __init__(self, employee: Employee, rating: float, feedback: str, bonus: int):
        """
        Initializes a Performance object.

        Args:
            employee: The Employee object this performance data belongs to.
            rating: The performance rating score.
            feedback: Qualitative feedback on performance.
            bonus: The bonus amount awarded.
        """
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus


# Global data stores for employees and their performance records
employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}

def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Retrieves and formats employee performance data using a template string.

    Args:
        emp_key (str): The identifier of the employee.
        template_str (str): The string containing the summary format with
                            placeholders like '{employee.name}' or
                            '{performance.rating}'.

    Returns:
        str: A string formatted with the employee's performance summary.

    Raises:
        ValueError: If the employee key is invalid, performance data is missing,
                    placeholders are invalid, restricted fields (e.g., feedback,
                    bonus) are accessed, or attributes are not found.
    """
    # Fetch employee and performance data; this also validates emp_key
    employee, performance = _fetch_employee_data(emp_key)
    
    # Generate the summary string using the fetched data and template
    return _generate_template_output(template_str, employee, performance)


# Module-level constant for restricted fields in the Performance object
RESTRICTED_PERFORMANCE_FIELDS = ["feedback", "bonus"]


def _fetch_employee_data(emp_key: str) -> tuple[Employee, Performance]:
    """
    Fetches employee and their corresponding performance data.

    Args:
        emp_key (str): The identifier of the employee.

    Returns:
        tuple[Employee, Performance]: A tuple containing the Employee and
                                      Performance objects.

    Raises:
        ValueError: If the employee key is not found in `employees` or
                    `performances` data.
    """
    if emp_key not in employees:
        raise ValueError(f"Invalid employee key: {emp_key}")
    if emp_key not in performances:
        # This check ensures that performance data exists for the given employee.
        # It assumes that every valid employee in this context must have performance data.
        raise ValueError(f"Performance data not found for employee key: {emp_key}")
    return employees[emp_key], performances[emp_key]


def _check_and_get_value(
    obj: object, obj_name: str, attr_name: str, placeholder_content: str
) -> str:
    """
    Checks for restricted fields and retrieves an attribute's value as a string.

    Args:
        obj (object): The object (Employee or Performance) to access.
        obj_name (str): The name of the object type ("employee" or "performance")
                        for error messages.
        attr_name (str): The name of the attribute to access.
        placeholder_content (str): The full placeholder string (e.g., "employee.name")
                                   for error messages.

    Returns:
        str: The string representation of the attribute's value.

    Raises:
        ValueError: If a restricted field is accessed, or if the attribute
                    does not exist on the object.
    """
    if obj_name == "performance":
        if attr_name in RESTRICTED_PERFORMANCE_FIELDS:
            raise ValueError(
                f"Access to restricted performance field '{attr_name}' in "
                f"placeholder {{{placeholder_content}}} is not allowed."
            )
        if attr_name == "employee":  # Prevent direct access to the nested Employee object
            raise ValueError(
                f"Accessing the 'employee' object via 'performance.employee' "
                f"placeholder {{{placeholder_content}}} is not allowed. "
                f"Use 'employee.attribute_name' directly."
            )

    if not hasattr(obj, attr_name):
        raise ValueError(
            f"Invalid attribute '{attr_name}' for {obj_name} in "
            f"placeholder {{{placeholder_content}}}"
        )
    return str(getattr(obj, attr_name))


def _generate_template_output(
    template_str: str, employee: Employee, performance: Performance
) -> str:
    """
    Generates the summary string by replacing placeholders with actual data.

    Args:
        template_str (str): The template string with placeholders.
        employee (Employee): The employee data object.
        performance (Performance): The performance data object for the employee.

    Returns:
        str: The template string with placeholders substituted with data.

    Raises:
        ValueError: If template contains unresolved or invalid placeholders,
                    or if errors occur during value retrieval (e.g., restricted
                    field access).
    """

    def replace_placeholder(match_obj: re.Match) -> str:
        """
        Callback function for re.sub to replace a single placeholder.
        It extracts object name and attribute name from the matched placeholder
        (e.g., "employee.name") and retrieves the corresponding value.
        """
        placeholder_content = match_obj.group(1)  # Content inside { }
        # Split "object.attribute" into object_name and attr_name
        obj_name, attr_name = placeholder_content.split('.', 1)

        if obj_name == "employee":
            return _check_and_get_value(
                employee, "employee", attr_name, placeholder_content
            )
        elif obj_name == "performance":
            return _check_and_get_value(
                performance, "performance", attr_name, placeholder_content
            )
        else:  # This case should ideally not be reached due to the main regex pattern
               # pragma: no cover
            raise ValueError(
                f"Invalid object type '{obj_name}' in placeholder: {{{placeholder_content}}}"
            )

    # Regex to find placeholders like {employee.attribute} or {performance.attribute}
    # \w+ matches one or more word characters (letters, numbers, underscore).
    regex_pattern = r'\{(employee\.\w+|performance\.\w+)\}'
    
    try:
        # Substitute all valid placeholders using the replace_placeholder callback
        formatted_string = re.sub(regex_pattern, replace_placeholder, template_str)
    except ValueError:  # Re-raise ValueErrors from _check_and_get_value
        raise

    # After substitution, check if any other types of placeholders remain.
    # This catches placeholders not matching the specific 'employee.attr' or
    # 'performance.attr' patterns, or any other malformed placeholders.
    remaining_placeholders_match = re.findall(r'\{([^}]+)\}', formatted_string)
    if remaining_placeholders_match:
        # Collect all unresolved placeholders for a comprehensive error message
        invalid_placeholders_str = [f"{{{p}}}" for p in remaining_placeholders_match]
        raise ValueError(
            f"Template contains unresolved or invalid placeholders: {invalid_placeholders_str}"
        )

    return formatted_string
