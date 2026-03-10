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

def _filter_employees_by_department(all_employees: list[Employee], department_name: str) -> list[Employee]:
    """Filters a list of employees by department."""
    return [emp for emp in all_employees if emp.department == department_name]

def _process_employee_template(employee: Employee, template: str, sensitive_fields: list[str]) -> str:
    """
    Processes a template for a single employee, handling sensitive data checks
    and replacing placeholders.
    Raises ValueError for disallowed access or processing errors.
    """
    
    # 1. Initial check for any sensitive field placeholder strings in the template
    for field in sensitive_fields:
        if f"{{employee.{field}}}" in template:
            raise ValueError(f"Access to sensitive field '{field}' is not allowed in template.")

    # 2. Define allowed attributes for this employee that can be used in the template
    allowed_data_for_template = {
        "name": employee.name,
        "position": employee.position,
        "department": employee.department
    }

    # 3. Parse all {employee.X} placeholders to ensure they are valid and not sensitive
    placeholders = re.findall(r"\{employee\.(\w+)\}", template)
    for placeholder_attribute in placeholders:
        if placeholder_attribute in sensitive_fields:
            # This check is defense-in-depth; check #1 should catch this.
            raise ValueError(f"Access to sensitive field '{placeholder_attribute}' via placeholder is not allowed.")
        if placeholder_attribute not in allowed_data_for_template: # Check against keys of allowed data
            raise ValueError(f"Access to unspecified or disallowed field '{placeholder_attribute}' is not allowed.")

    # 4. Perform the replacement
    processed_template = template
    try:
        for attr_key, attr_value in allowed_data_for_template.items():
            processed_template = processed_template.replace(f"{{employee.{attr_key}}}", str(attr_value))
        
        # Sanity check: After replacement, ensure no {employee.X} placeholders that were
        # supposed to be processed (i.e., in allowed_data_for_template) remain.
        final_check_placeholders = re.findall(r"\{employee\.(\w+)\}", processed_template)
        problematic_remaining = [p for p in final_check_placeholders if p in allowed_data_for_template]
        if problematic_remaining:
             raise ValueError(f"Template processing error: Placeholder for allowed field '{problematic_remaining[0]}' remained after replacement.")

    except Exception as e: 
        raise ValueError(f"Error during template string replacement for employee {employee.name}: {str(e)}")

    return processed_template

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
    # Sensitive fields list remains here as it's a policy for this function.
    sensitive_fields = ["salary", "personal_email"]

    # Use the helper function to filter employees from the global 'employees' list.
    department_employees = _filter_employees_by_department(employees, department)

    if not department_employees:
        return "" # As per original logic for empty department

    for emp in department_employees:
        try:
            # Use the helper function to process the template for each employee
            processed_entry = _process_employee_template(emp, template, sensitive_fields)
            directory_entries.append(processed_entry)
        except ValueError:
            # Propagate ValueError from helper, as per requirement to raise ValueError on any error.
            raise
        # Other specific exceptions from _process_employee_template are converted to ValueError
        # within that function, so only ValueError is expected here.

    return "\n".join(directory_entries)
