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
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
    Employee("David Brown", "HR Specialist", "HR", 70000, "david@personal.com"),
]

RESTRICTED_FIELDS = ["salary", "personal_email"]

def generate_team_roster(division_name: str, roster_template: str) -> str:
    """
    Compiles an employee directory based on a specified template format.

    Args:
        division_name: The department name to filter by.
        roster_template: The string template to populate with employee details.
                         Placeholders like '{employee.name}' and '{employee.position}'
                         will be replaced.

    Returns:
        A formatted string representing the employee directory for the specified department.

    Raises:
        ValueError: If the function attempts to access restricted fields like salary
                    or personal email, or if an unexpected error occurs during
                    template population (e.g., invalid attribute).
    """
    # Check for restricted fields in the template
    placeholders = re.findall(r"\{employee\.(\w+)\}", roster_template)
    for field in placeholders:
        if field in RESTRICTED_FIELDS:
            raise ValueError(f"Access to restricted field '{field}' is not allowed.")

    department_employees = [emp for emp in employees if emp.department == division_name]
    
    if not department_employees:
        return ""

    formatted_roster_entries = []
    for employee in department_employees:
        try:
            current_roster_entry = roster_template
            for placeholder in placeholders:
                # Ensure the attribute exists before trying to access it
                if not hasattr(employee, placeholder):
                    raise ValueError(f"Employee object does not have attribute '{placeholder}'.")
                value = getattr(employee, placeholder)
                current_roster_entry = current_roster_entry.replace(f"{{employee.{placeholder}}}", str(value))
            formatted_roster_entries.append(current_roster_entry)
        except AttributeError as e:
            # This might catch cases where getattr itself fails for some reason,
            # though the hasattr check should prevent most of these.
            raise ValueError(f"An unexpected error occurred while accessing employee data: {e}")
        except Exception as e:
            # Catch any other unexpected errors during formatting
            raise ValueError(f"An unexpected error occurred: {e}")
            
    return "\n".join(formatted_roster_entries)

if __name__ == '__main__':
    # Example Usage:
    template_it = "Name: {employee.name}, Position: {employee.position}"
    division_it = "IT"
    print(f"--- {division_it} Roster ---")
    try:
        roster_it = generate_team_roster(division_it, template_it)
        print(roster_it)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- HR Roster (Name Only) ---")
    template_hr = "HR Member: {employee.name}"
    division_hr = "HR"
    try:
        roster_hr = generate_team_roster(division_hr, template_hr)
        print(roster_hr)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access restricted field (Salary) ---")
    template_restricted_salary = "Name: {employee.name}, Salary: {employee.salary}"
    try:
        roster_restricted = generate_team_roster(division_it, template_restricted_salary)
        print(roster_restricted)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access restricted field (Personal Email) ---")
    template_restricted_email = "Name: {employee.name}, Email: {employee.personal_email}"
    try:
        roster_restricted = generate_team_roster(division_it, template_restricted_email)
        print(roster_restricted)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access non-existent field ---")
    template_non_existent = "Name: {employee.name}, Nickname: {employee.nickname}"
    try:
        roster_non_existent = generate_team_roster(division_it, template_non_existent)
        print(roster_non_existent)
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n--- Roster for non-existent department ---")
    template_marketing = "Name: {employee.name}, Position: {employee.position}"
    division_marketing = "Marketing"
    try:
        roster_marketing = generate_team_roster(division_marketing, template_marketing)
        if not roster_marketing:
            print(f"No employees found for {division_marketing} department.")
        else:
            print(roster_marketing)
    except ValueError as e:
        print(f"Error: {e}")
