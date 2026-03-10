import logging
import re
from typing import List, Set


logger = logging.getLogger(__name__)


class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email


employees: List[Employee] = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
]


# Constants for template parsing and validation
RESTRICTED_FIELDS: Set[str] = {"salary", "personal_email"}
PLACEHOLDER_PATTERN = re.compile(r"\{employee\.([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _extract_template_fields(list_template: str) -> Set[str]:
    """Extract all employee attribute names referenced by the template."""
    return set(PLACEHOLDER_PATTERN.findall(list_template))


def _validate_no_restricted_fields(fields: Set[str], team_name: str, template: str) -> None:
    """Ensure the template does not reference restricted fields. Log and raise on violation."""
    restricted_used = sorted(RESTRICTED_FIELDS.intersection(fields))
    if restricted_used:
        logger.warning(
            "Restricted field(s) referenced in template. team=%s fields=%s template=%r",
            team_name,
            restricted_used,
            template,
        )
        raise ValueError(
            f"Access to restricted field(s) {', '.join(restricted_used)} is not allowed."
        )


def _filter_employees_by_department(team_name: str, roster: List[Employee]) -> List[Employee]:
    """Return employees whose department matches the provided team name."""
    return [emp for emp in roster if emp.department == team_name]


def _render_template_for_employee(list_template: str, emp: Employee) -> str:
    """Fill the template for a single employee, enforcing field restrictions."""

    def replacer(match: re.Match) -> str:
        attr = match.group(1)

        # Guard: Restricted field access attempt
        if attr in RESTRICTED_FIELDS:
            logger.warning(
                "Attempted restricted field access during rendering. employee=%s field=%s department=%s",
                emp.name,
                attr,
                emp.department,
            )
            raise ValueError(f"Access to restricted field '{attr}' is not allowed.")

        # Guard: Unknown field reference
        if not hasattr(emp, attr):
            logger.error(
                "Unknown field referenced in template. employee=%s field=%s department=%s",
                emp.name,
                attr,
                emp.department,
            )
            raise ValueError(f"Unknown field '{attr}' in template.")

        return str(getattr(emp, attr))

    return PLACEHOLDER_PATTERN.sub(replacer, list_template)


def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Build an employee directory string for a specified department based on a template.

    Args:
        team_name: The department name to filter by.
        list_template: A string template containing placeholders such as
                       '{employee.name}' and '{employee.position}'.

    Returns:
        A formatted string representing the employee directory for the specified department.

    Raises:
        ValueError: If the template attempts to access restricted fields (salary, personal_email),
                    if a referenced field does not exist, or if an unexpected error occurs.
    """
    try:
        logger.info("Building worker list. team=%s", team_name)

        # Parse and validate template fields
        fields_in_template = _extract_template_fields(list_template)
        _validate_no_restricted_fields(fields_in_template, team_name, list_template)

        # Filter employees (guard clause for empty result)
        filtered = _filter_employees_by_department(team_name, employees)
        if not filtered:
            logger.info("No employees found for team. team=%s", team_name)
            return ""

        logger.debug(
            "Employees matched. team=%s count=%d names=%s",
            team_name,
            len(filtered),
            [emp.name for emp in filtered],
        )

        # Render each employee line
        result_lines: List[str] = [
            _render_template_for_employee(list_template, emp) for emp in filtered
        ]

        result = "\n".join(result_lines)
        logger.info("Worker list built successfully. team=%s lines=%d", team_name, len(result_lines))
        return result
    except ValueError:
        # Re-raise explicit ValueErrors
        raise
    except Exception as exc:
        logger.exception(
            "Unexpected error while building worker list. team=%s", team_name
        )
        raise ValueError("An unexpected error occurred while building the worker list.") from exc
