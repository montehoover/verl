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

def create_employee_directory(department: str, template: str) -> str:
    """
    Generates a directory string for a given department using a template.

    Args:
        department: The name of the department.
        template: The directory template string with placeholders.

    Returns:
        The generated directory string.

    Raises:
        ValueError: If any error occurs or if access to sensitive
                    information (salary, personal_email) is attempted.
    """
    directory_entries = []
    sensitive_fields = ["salary", "personal_email"]

    department_employees = [emp for emp in employees if emp.department == department]

    if not department_employees:
        return "" # Or raise ValueError("No employees found in this department")

    for emp in department_employees:
        entry = template
        # Basic placeholder replacement - a more robust solution might use regex
        # to find all placeholders like {employee.attribute}
        
        # Check for sensitive fields in the template
        for field in sensitive_fields:
            if f"{{employee.{field}}}" in template:
                raise ValueError(f"Access to sensitive field '{field}' is not allowed.")
            # Also check for direct attribute access if template evaluation was more complex
            # For string.format or f-strings, this check is more about the placeholder name
            # than actual attribute access during formatting.

        try:
            # Replace allowed placeholders
            # A more robust way would be to parse placeholders and use getattr
            # For simplicity, we'll do direct replacement for known safe fields.
            # This assumes placeholders are exactly like {employee.name} or {employee.position}
            
            # A safer way to format, ensuring only allowed attributes are accessed.
            # We can create a dictionary of allowed attributes for formatting.
            allowed_attrs = {
                "name": emp.name,
                "position": emp.position,
                "department": emp.department # Department might be redundant if already filtered
            }
            
            # Check if template tries to access non-allowed attributes
            # This requires parsing the template for placeholders
            import re
            placeholders = re.findall(r"\{employee\.(\w+)\}", template)
            for ph_attr in placeholders:
                if ph_attr not in allowed_attrs and ph_attr not in sensitive_fields:
                    # This case means an attribute is requested that is not explicitly allowed
                    # and not explicitly sensitive. Depending on policy, this could be an error
                    # or ignored. For now, let's assume only 'name', 'position', 'department' are allowed.
                    if ph_attr not in ["name", "position", "department"]:
                         raise ValueError(f"Access to unspecified field '{ph_attr}' is not allowed.")
                elif ph_attr in sensitive_fields: # Double check, already done above but good for safety
                    raise ValueError(f"Access to sensitive field '{ph_attr}' is not allowed.")

            # Perform the replacement using the allowed attributes
            # This is a bit tricky with str.format if placeholders are dynamic.
            # A simpler approach for this problem might be direct string replacement
            # if the template format is strictly {employee.attribute}.

            current_entry = template
            current_entry = current_entry.replace("{employee.name}", emp.name)
            current_entry = current_entry.replace("{employee.position}", emp.position)
            current_entry = current_entry.replace("{employee.department}", emp.department)
            
            # If after replacement, some {employee.X} placeholders remain for sensitive fields,
            # it means they were not caught by the initial check (e.g. typo in check).
            # This is a fallback.
            for field in sensitive_fields:
                if f"{{employee.{field}}}" in current_entry: # Check if it's still in the template string
                    # This check is somewhat redundant if the first check is robust
                    pass


            directory_entries.append(current_entry)

        except AttributeError as e:
            # This might catch attempts to access attributes not present on Employee
            raise ValueError(f"Invalid attribute in template: {e}")
        except Exception as e:
            # Catch any other formatting errors
            raise ValueError(f"Error processing template: {e}")

    return "\n".join(directory_entries)
