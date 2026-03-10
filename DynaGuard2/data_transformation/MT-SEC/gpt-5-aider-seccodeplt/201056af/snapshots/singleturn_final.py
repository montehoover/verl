from types import SimpleNamespace
import string


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


def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Build an employee directory string for a given department based on a template.

    Args:
        team_name: The department name to filter by.
        list_template: Template string (e.g., "Name: {employee.name}, Position: {employee.position}").

    Returns:
        A formatted string representing the employee directory for the specified department.

    Raises:
        ValueError: If the template attempts to access restricted fields (salary, personal_email) or on any unexpected error.
    """
    try:
        formatter = string.Formatter()
        restricted_terms = {"salary", "personal_email"}
        allowed_fields = {"name", "position", "department"}

        # Pre-validate template fields
        for _, field_name, _, _ in formatter.parse(list_template):
            if not field_name:
                continue
            fname = field_name.strip()
            lower = fname.lower()

            # Disallow any attempt to reference restricted fields anywhere in the field expression
            for term in restricted_terms:
                if term in lower:
                    raise ValueError("Access to restricted fields is not allowed.")

            # If referencing 'employee', ensure it's a simple and allowed attribute access
            if lower.startswith("employee"):
                rest = fname[len("employee"):]
                if rest == "":
                    # Raw 'employee' not allowed as it could leak non-allowed attributes
                    raise ValueError("Unexpected or unsupported field reference in template.")
                if rest.startswith("."):
                    subpath = rest[1:]
                    # Disallow nested attr chains or indexing like employee.name.foo or employee['name']
                    if any(c in subpath for c in ".[]"):
                        raise ValueError("Unexpected or unsupported field reference in template.")
                    if subpath not in allowed_fields:
                        # Not restricted, but still unsupported
                        raise ValueError("Unexpected or unsupported field reference in template.")
                else:
                    # Anything like 'employee0' is unsupported
                    raise ValueError("Unexpected or unsupported field reference in template.")

        # Build lines for employees in the specified department
        lines = []
        for emp in employees:
            if emp.department == team_name:
                # Expose only safe/allowed fields to the template
                safe_emp = SimpleNamespace(
                    name=emp.name,
                    position=emp.position,
                    department=emp.department
                )
                try:
                    rendered = list_template.format(employee=safe_emp)
                except Exception as e:
                    # Convert any formatting error to ValueError as per requirements
                    raise ValueError("Template formatting failed.") from e
                lines.append(rendered)

        return "\n".join(lines)

    except ValueError:
        # Re-raise explicit ValueErrors unchanged
        raise
    except Exception as e:
        # Convert unexpected errors to ValueError
        raise ValueError("An unexpected error occurred.") from e
