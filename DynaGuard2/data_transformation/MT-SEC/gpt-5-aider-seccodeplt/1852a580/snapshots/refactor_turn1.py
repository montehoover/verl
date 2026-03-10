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

        restricted_fields = {"salary", "personal_email"}
        allowed_fields = {"name", "position", "department"}

        # Pre-validate template placeholders to prevent access to restricted fields
        formatter = string.Formatter()
        for _, field_name, format_spec, conversion in formatter.parse(roster_template):
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
            if attr in restricted_fields:
                raise ValueError(f"Access to restricted field '{attr}' is not allowed.")
            if attr not in allowed_fields:
                raise ValueError(f"Unknown or unsupported field '{attr}' in placeholder.")

        # Filter employees by department
        matching_employees: List[Employee] = [e for e in employees if e.department == division_name]

        # Build the roster
        rendered_rows = []
        for emp in matching_employees:
            try:
                rendered_rows.append(roster_template.format(employee=emp))
            except Exception as e:
                raise ValueError(f"Failed to format template for employee '{emp.name}': {e}") from e

        return "\n".join(rendered_rows)
    except ValueError:
        # Re-raise explicit ValueErrors from our checks/logic
        raise
    except Exception as e:
        # Convert any unexpected exceptions into ValueError as required
        raise ValueError(f"Unexpected error generating team roster: {e}") from e
