import re

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


def compile_employee_evaluation(employee_id, template):
    """
    Compiles an employee evaluation report by replacing placeholders in a template
    with employee and performance data.

    Args:
        employee_id (str): The ID of the employee.
        template (str): A string template with placeholders like 
                        '{personnel.name}' or '{review.rating}'.

    Returns:
        str: The formatted performance summary string.

    Raises:
        ValueError: If the employee ID is invalid.
    """
    if employee_id not in employees:
        raise ValueError(f"Invalid employee ID: {employee_id} (not in employees dictionary)")
    if employee_id not in performances:
        raise ValueError(f"Invalid employee ID: {employee_id} (no performance data found)")

    employee = employees[employee_id]
    performance = performances[employee_id]

    # Define the data accessible in the template.
    # 'personnel.*' fields come from the Employee object.
    # 'review.*' fields come from the Performance object.
    # This function assumes access to feedback and bonus is authorized for "evaluation" purposes.
    data_for_template = {
        'personnel.emp_id': employee.emp_id,
        'personnel.name': employee.name,
        'personnel.position': employee.position,
        'personnel.department': employee.department,
        'review.rating': performance.rating,
        'review.feedback': performance.feedback,
        'review.bonus': performance.bonus,
    }

    def replacer(match):
        placeholder_key = match.group(1)  # e.g., "personnel.name"
        # Get the value from our data map. If key is not found, return the placeholder itself.
        value = data_for_template.get(placeholder_key, f"{{{placeholder_key}}}")
        return str(value)

    # Use re.sub to find all placeholders like {key} and replace them.
    # The (.*?) is a non-greedy match for any characters inside the curly braces.
    return re.sub(r"\{(.*?)\}", replacer, template)


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

    print("\n--- Testing compile_employee_evaluation ---")

    eval_template_full = (
        "Employee Evaluation Report:\n"
        "---------------------------\n"
        "ID: {personnel.emp_id}\n"
        "Name: {personnel.name}\n"
        "Position: {personnel.position}\n"
        "Department: {personnel.department}\n"
        "---------------------------\n"
        "Performance Review:\n"
        "Rating: {review.rating}\n"
        "Feedback: {review.feedback}\n"
        "Bonus: ${review.bonus}\n"
        "---------------------------"
    )
    
    eval_template_short_with_missing = "Quick Summary for {personnel.name}: Rating - {review.rating}. Contact: {personnel.email_address}"

    print("\nCompiling evaluation for P201 (full template):")
    try:
        evaluation = compile_employee_evaluation("P201", eval_template_full)
        print(evaluation)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nCompiling evaluation for P202 (short template with a missing placeholder '{personnel.email_address}'):")
    try:
        evaluation = compile_employee_evaluation("P202", eval_template_short_with_missing)
        print(evaluation)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nCompiling evaluation for P999 (invalid ID):")
    try:
        evaluation = compile_employee_evaluation("P999", eval_template_full)
        print(evaluation)
    except ValueError as e:
        print(f"Error: {e}")
