import re
import logging

# Configure basic logging
# In a real application, this might be configured at a higher level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]

ALLOWED_FIELDS = {'name', 'position', 'department'}
RESTRICTED_FIELDS = {'salary', 'personal_email'}


def _filter_employees_by_department(
    all_employees: list[Employee], division_name: str
) -> list[Employee]:
    """Filters a list of employees by their department."""
    return [e for e in all_employees if e.department == division_name]


def _format_roster_for_employees(
    department_employees: list[Employee],
    roster_template: str,
    placeholders_in_template: list[str],
) -> str:
    """Formats a roster string for a list of employees using a template."""
    if not department_employees:
        return ""

    final_roster_parts = []
    for emp in department_employees:
        current_employee_roster = roster_template
        for field_name in placeholders_in_template:
            placeholder_tag = f"{{employee.{field_name}}}"
            try:
                value = getattr(emp, field_name)
            except AttributeError:
                # This case implies an inconsistency between ALLOWED_FIELDS
                # and actual Employee attributes, as validation in the caller
                # should prevent unknown fields from reaching here.
                raise ValueError(
                    f"Error accessing attribute '{field_name}' for employee '{emp.name}'. "
                    "This may indicate an inconsistency in field definitions or ALLOWED_FIELDS."
                )
            current_employee_roster = current_employee_roster.replace(
                placeholder_tag, str(value)
            )
        final_roster_parts.append(current_employee_roster)

    return "\n".join(final_roster_parts)


def generate_team_roster(division_name: str, roster_template: str) -> str:
    """
    Compiles an employee directory for a specified department based on a template format.

    Args:
        division_name: The department name to filter by.
        roster_template: The string template to populate with employee details.
                         Placeholders like '{employee.name}' are supported.

    Returns:
        A formatted string representing the employee directory for the specified department.
        Each employee's record is separated by a newline. If no employees are found
        in the specified department, an empty string is returned.

    Raises:
        ValueError: If the template attempts to access restricted fields (salary, personal_email),
                    refers to unknown fields, or if an unexpected error occurs during processing.
    """
    logger.info(
        f"Generating team roster for division: '{division_name}' with template: '{roster_template}'"
    )
    try:
        # Validate template placeholders by checking all {employee.field_name} occurrences
        placeholders_in_template = re.findall(r'{employee\.(\w+)}', roster_template)
        for field_name in placeholders_in_template:
            if field_name in RESTRICTED_FIELDS:
                logger.error(f"Template validation failed: Attempt to access restricted field '{field_name}'.")
                raise ValueError(f"Attempt to access restricted field: {field_name}")
            if field_name not in ALLOWED_FIELDS:
                logger.error(f"Template validation failed: Attempt to access unknown or not permitted field '{field_name}'.")
                raise ValueError(f"Attempt to access unknown or not permitted field: {field_name}")
        
        logger.debug(f"Validated placeholders in template: {placeholders_in_template}")

        # Step 1: Filter employees by department
        # The global 'employees' list is passed here.
        department_employees = _filter_employees_by_department(employees, division_name)
        logger.info(f"Found {len(department_employees)} employees in department '{division_name}'.")
        
        # Step 2: Format the roster for the filtered employees
        formatted_roster = _format_roster_for_employees(
            department_employees,
            roster_template,
            placeholders_in_template
        )
        
        logger.info(f"Successfully generated roster for division '{division_name}'.")
        # For potentially large rosters, consider logging a summary or hash instead of the full content.
        # For this example, we log a snippet or the full roster if short.
        roster_log_display = (formatted_roster[:200] + '...') if len(formatted_roster) > 200 else formatted_roster
        logger.debug(f"Formatted Roster Snippet:\n{roster_log_display}")
        
        return formatted_roster

    except ValueError as ve:  # Re-raise ValueErrors explicitly raised (e.g., validation, formatting)
        logger.error(f"ValueError during roster generation for '{division_name}': {ve}")
        raise
    except Exception as e:  # Catch any other unexpected exception and wrap it in ValueError
        logger.exception(f"Unexpected error during roster generation for '{division_name}': {e}")
        raise ValueError(f"An unexpected error occurred during roster generation: {str(e)}") from e
