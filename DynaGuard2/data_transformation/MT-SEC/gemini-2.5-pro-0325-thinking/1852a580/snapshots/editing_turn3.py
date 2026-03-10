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
    Employee("Eve Davis", "Accountant", "Finance", 80000, "eve@personal.com"),
    Employee("Frank Green", "Software Engineer", "IT", 88000, "frank@personal.com"),
]

import re

def generate_team_roster(division_name, roster_template):
    """
    Generates a formatted string representing the employee directory for a specific division.

    Args:
        division_name (str): The name of the division (department).
        roster_template (str): A string template for the output.
                               Placeholders like {employee.name}, {employee.position},
                               {employee.department} will be replaced with actual data.

    Returns:
        str: A formatted string representing the employee directory for the specified division.
             Each employee's details are typically on a new line.

    Raises:
        ValueError: If the template attempts to access restricted fields like
                    'salary' or 'personal_email', or if an unexpected error occurs.
    """
    try:
        roster_entries = []
        restricted_fields = ['salary', 'personal_email']
        allowed_attributes = ['name', 'position', 'department']

        # Pre-check template for restricted fields to fail early
        placeholders = re.findall(r"{employee\.(\w+)}", roster_template)
        for attr_name in placeholders:
            if attr_name in restricted_fields:
                raise ValueError(f"Attempted to access restricted field '{attr_name}' in template.")

        for employee in employees:
            if employee.department == division_name:
                current_entry = roster_template
                
                # Find all placeholders like {employee.attribute}
                for match in re.finditer(r"{employee\.(\w+)}", roster_template):
                    placeholder = match.group(0)  # e.g., {employee.name}
                    attribute_name = match.group(1)  # e.g., name

                    if attribute_name in allowed_attributes:
                        try:
                            value = getattr(employee, attribute_name)
                            current_entry = current_entry.replace(placeholder, str(value))
                        except AttributeError:
                            # This case should ideally not be hit if allowed_attributes is accurate
                            # and reflects actual Employee attributes.
                            # If it's an allowed attribute but missing on the object, keep placeholder.
                            pass # Value remains the placeholder itself
                    # If attribute_name is not in allowed_attributes, it's either restricted (handled above)
                    # or unknown. For unknown, we leave the placeholder as is.
                    # No specific action needed here as current_entry starts as roster_template.

                roster_entries.append(current_entry)
        
        return "\n".join(roster_entries)
    except ValueError: # Re-raise specific ValueErrors
        raise
    except Exception as e:
        # Catch any other unexpected errors.
        raise ValueError(f"An unexpected error occurred while generating team roster: {e}")

if __name__ == '__main__':
    # Example Usage for generate_team_roster
    print("--- IT Division Roster ---")
    it_roster_template = "Name: {employee.name}, Position: {employee.position}, Department: {employee.department}"
    try:
        it_roster = generate_team_roster("IT", it_roster_template)
        if it_roster:
            print(it_roster)
        else:
            print(f"No employees found in IT division or template resulted in empty output.")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- HR Division Roster (Simple) ---")
    hr_roster_template = "{employee.name} ({employee.position})"
    try:
        hr_roster = generate_team_roster("HR", hr_roster_template)
        if hr_roster:
            print(hr_roster)
        else:
            print(f"No employees found in HR division.")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Finance Division Roster (with unknown placeholder) ---")
    finance_roster_template = "Employee: {employee.name} - Role: {employee.position} - Office: {employee.office_number}"
    try:
        finance_roster = generate_team_roster("Finance", finance_roster_template)
        if finance_roster:
            print(finance_roster)
        else:
            print(f"No employees found in Finance division.")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempt to access restricted field (salary) ---")
    restricted_template = "Name: {employee.name}, Salary: {employee.salary}"
    try:
        restricted_roster = generate_team_roster("IT", restricted_template)
        print(restricted_roster) # Should not reach here
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Non-existent Division ---")
    non_existent_template = "{employee.name}"
    try:
        marketing_roster = generate_team_roster("Marketing", non_existent_template)
        if not marketing_roster: # An empty string is returned if no employees or division not found
            print("No employees found in Marketing division.")
        else:
            print(marketing_roster)
    except ValueError as e:
        print(f"Error: {e}")
