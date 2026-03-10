import re

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
    try:
        # Validate template placeholders by checking all {employee.field_name} occurrences
        placeholders_in_template = re.findall(r'{employee\.(\w+)}', roster_template)
        for field_name in placeholders_in_template:
            if field_name in RESTRICTED_FIELDS:
                raise ValueError(f"Attempt to access restricted field: {field_name}")
            if field_name not in ALLOWED_FIELDS:
                raise ValueError(f"Attempt to access unknown or not permitted field: {field_name}")

        department_employees = [e for e in employees if e.department == division_name]
        
        if not department_employees:
            return ""  # No employees in the department

        final_roster_parts = []
        for emp in department_employees:
            current_employee_roster = roster_template
            # Replace validated placeholders with actual data for the current employee
            for field_name in placeholders_in_template:
                placeholder_tag = f"{{employee.{field_name}}}" # e.g., "{employee.name}"
                try:
                    # Ensure getattr is only used for fields confirmed to be in ALLOWED_FIELDS
                    value = getattr(emp, field_name)
                except AttributeError:
                    # This case should ideally not be reached if ALLOWED_FIELDS accurately
                    # reflects Employee attributes and validation is correct.
                    # This acts as a safeguard for internal consistency.
                    raise ValueError(f"Error accessing attribute '{field_name}' for employee '{emp.name}'. "
                                     "This may indicate an inconsistency in field definitions.")
                
                current_employee_roster = current_employee_roster.replace(placeholder_tag, str(value))
            
            final_roster_parts.append(current_employee_roster)
        
        return "\n".join(final_roster_parts)

    except ValueError:  # Re-raise ValueErrors that were explicitly raised within this function
        raise
    except Exception as e:  # Catch any other unexpected exception and wrap it in ValueError
        # This fulfills the requirement "if an unexpected error occurs"
        raise ValueError(f"An unexpected error occurred during roster generation: {str(e)}") from e
