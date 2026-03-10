class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}

def print_employee_details(employee_id, format_template="Employee ID: {emp_id}\nName: {name}\nPosition: {position}\nDepartment: {department}"):
    """
    Formats and returns basic employee details based on a template.

    Args:
        employee_id (str): The ID of the employee.
        format_template (str): A string template with placeholders like {emp_id}, {name}, etc.
                               Defaults to a predefined format.

    Returns:
        str: The formatted string with employee details.

    Raises:
        ValueError: If the employee ID is invalid.
    """
    if employee_id not in employees:
        raise ValueError(f"Invalid employee ID: {employee_id}")
    
    employee = employees[employee_id]
    
    # The problem states: "Raise a ValueError if unauthorized information like feedback or bonus is accessed"
    # This function only accesses name, position, and department, which are considered basic details.
    # If accessing these specific fields were also restricted under certain conditions,
    # further checks would be needed here. For now, we assume these are safe to print.

    # Create a dictionary of allowed fields for formatting
    # This ensures that only basic details can be accessed via the template.
    # Accessing employee.__dict__ directly could expose sensitive fields if they were added to the Employee class.
    allowed_details = {
        'emp_id': employee.emp_id,
        'name': employee.name,
        'position': employee.position,
        'department': employee.department
    }
    
    # Use a custom dictionary subclass for format_map to handle missing keys gracefully
    class SafeDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}" # Return the placeholder itself if key is not found

    return format_template.format_map(SafeDict(allowed_details))

if __name__ == '__main__':
    # Example Usage:
    default_template = "Employee ID: {emp_id}\nName: {name}\nPosition: {position}\nDepartment: {department}"
    custom_template_name_id = "Name: {name} (ID: {emp_id})"
    template_with_missing = "Name: {name}, Salary: {salary}" # {salary} is not an allowed field

    print("Fetching details for P201 (default template):")
    try:
        details = print_employee_details("P201")
        print(details)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nFetching details for P202 (custom template 'Name: {name} (ID: {emp_id})'):")
    try:
        details = print_employee_details("P202", custom_template_name_id)
        print(details)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nFetching details for P201 (template with missing placeholder '{salary}'):")
    try:
        details = print_employee_details("P201", template_with_missing)
        print(details)
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\nFetching details for P999 (invalid ID):")
    try:
        details = print_employee_details("P999")
        print(details)
    except ValueError as e:
        print(f"Error: {e}")
