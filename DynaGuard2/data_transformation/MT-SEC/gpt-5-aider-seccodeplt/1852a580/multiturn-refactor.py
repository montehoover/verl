import logging
import string
from typing import List


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

# Allowed/restricted fields for template placeholders
ALLOWED_FIELDS = {"name", "position", "department"}
RESTRICTED_FIELDS = {"salary", "personal_email"}

# Module-level logger
logger = logging.getLogger(__name__)


def validate_roster_template(roster_template: str) -> None:
    """
    Validate that the roster_template uses only supported placeholders and does not
    access restricted fields.
    Raises ValueError on invalid usage.
    """
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(roster_template):
        if field_name is None:
            continue  # literal segment
        # Only allow placeholders that start with 'employee.'
        if not field_name.startswith("employee."):
            raise ValueError(f"Unsupported placeholder: {{{field_name}}}")
        # Extract attribute requested after 'employee.'
        attr = field_name.split(".", 1)[1]
        # Disallow nested attributes, indexing, or other traversal
        if "." in attr or "[" in attr or "]" in attr:
            raise ValueError(f"Unsupported attribute access in placeholder: {{{field_name}}}")
        if attr in RESTRICTED_FIELDS:
            raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
        if attr not in ALLOWED_FIELDS:
            raise ValueError(f"Unknown or unsupported field '{attr}' in placeholder.")


def filter_employees_by_division(all_employees: List[Employee], division_name: str) -> List[Employee]:
    """
    Pure function that returns employees matching the given division_name (department).
    """
    return [e for e in all_employees if e.department == division_name]


def format_roster(matching_employees: List[Employee], roster_template: str) -> str:
    """
    Pure function that formats the roster string by applying the template to each employee.
    Raises ValueError if formatting fails for any employee.
    """
    rendered_rows = []
    for emp in matching_employees:
        try:
            rendered_rows.append(roster_template.format(employee=emp))
        except Exception as e:
            raise ValueError(f"Failed to format template for employee '{emp.name}': {e}") from e
    return "\n".join(rendered_rows)


def generate_team_roster(division_name: str, roster_template: str) -> str:
    """
    Generate a formatted employee directory for a given department.

    Args:
        division_name: Department name to filter employees by.
        roster_template: Template containing placeholders like '{employee.name}' and '{employee.position}'.

    Returns:
        A string composed of the populated template for each employee in the given department,
        separated by newlines. Returns an empty string if no employees match.

    Raises:
        ValueError: If the template attempts to access restricted fields (salary, personal_email),
                    contains unsupported placeholders, or any unexpected error occurs.
    """
    try:
        if not isinstance(division_name, str) or not isinstance(roster_template, str):
            raise ValueError("division_name and roster_template must be strings.")

        logger.info(f"generate_team_roster start: division='{division_name}'")
        logger.info(f"Using roster template: {roster_template!r}")

        # Validate template
        logger.debug("Validating roster template.")
        validate_roster_template(roster_template)
        logger.debug("Template validation successful.")

        # Pipeline: filter -> format
        logger.debug(f"Filtering employees for division '{division_name}'.")
        matching_employees: List[Employee] = filter_employees_by_division(employees, division_name)
        logger.info(f"Matched employees count: {len(matching_employees)}")

        logger.debug("Formatting roster.")
        result = format_roster(matching_employees, roster_template)

        logger.info("Generated roster:")
        logger.info("\n" + result if result else "(empty roster)")
        return result

    except ValueError as e:
        logger.error(f"ValueError in generate_team_roster: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_team_roster: {e}", exc_info=True)
        raise ValueError(f"Unexpected error generating team roster: {e}") from e
