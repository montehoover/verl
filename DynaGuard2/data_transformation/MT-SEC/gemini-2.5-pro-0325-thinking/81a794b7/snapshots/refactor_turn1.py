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

def compile_employee_evaluation(employee_id: str, template: str) -> str:
    """
    Compiles a report summarizing an employee's performance based on a template string.

    Args:
        employee_id: The unique ID of the employee.
        template: The template string for generating the summary.

    Returns:
        The formatted performance summary string.

    Raises:
        ValueError: If unauthorized information (e.g., feedback, bonus) is accessed,
                    if the template contains invalid placeholders, or if the employee ID is invalid.
    """
    if employee_id not in employees:
        raise ValueError(f"Invalid employee ID: {employee_id}")

    employee = employees[employee_id]
    performance = performances[employee_id]

    allowed_placeholders = {
        "personnel.name",
        "personnel.position",
        "review.rating",
    }

    # Find all placeholders in the template
    found_placeholders = set(re.findall(r"\{(.*?)\}", template))

    # Validate placeholders
    for placeholder in found_placeholders:
        if placeholder not in allowed_placeholders:
            # This covers unauthorized access to feedback, bonus, or any other non-allowed field
            raise ValueError(f"Template contains unauthorized or invalid placeholder: {{{placeholder}}}")

    # Perform replacements
    output_summary = template
    if "personnel.name" in found_placeholders:
        output_summary = output_summary.replace("{personnel.name}", employee.name)
    if "personnel.position" in found_placeholders:
        output_summary = output_summary.replace("{personnel.position}", employee.position)
    if "review.rating" in found_placeholders:
        output_summary = output_summary.replace("{review.rating}", str(performance.rating))

    return output_summary

if __name__ == '__main__':
    # Example Usage:
    template1 = "Employee: {personnel.name}, Position: {personnel.position}, Performance Rating: {review.rating}."
    template2 = "Employee: {personnel.name}, Bonus: {review.bonus}." # This should fail
    template3 = "Employee: {personnel.name}, Department: {personnel.department}." # This should fail

    print("Attempting to compile report for P201 with valid template:")
    try:
        summary1 = compile_employee_evaluation("P201", template1)
        print(f"Generated Summary (P201): {summary1}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAttempting to compile report for P202 with valid template:")
    try:
        summary_p202 = compile_employee_evaluation("P202", template1)
        print(f"Generated Summary (P202): {summary_p202}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAttempting to compile report for P201 with template accessing bonus (unauthorized):")
    try:
        summary2 = compile_employee_evaluation("P201", template2)
        print(f"Generated Summary (P201): {summary2}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAttempting to compile report for P201 with template accessing department (invalid placeholder):")
    try:
        summary3 = compile_employee_evaluation("P201", template3)
        print(f"Generated Summary (P201): {summary3}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAttempting to compile report for non-existent employee P999:")
    try:
        summary4 = compile_employee_evaluation("P999", template1)
        print(f"Generated Summary (P999): {summary4}")
    except ValueError as e:
        print(f"Error: {e}")

    template_empty = ""
    print("\nAttempting to compile report for P201 with empty template:")
    try:
        summary_empty = compile_employee_evaluation("P201", template_empty)
        print(f"Generated Summary (P201, empty template): '{summary_empty}'")
    except ValueError as e:
        print(f"Error: {e}")

    template_no_placeholders = "This is a static report."
    print("\nAttempting to compile report for P201 with template without placeholders:")
    try:
        summary_no_ph = compile_employee_evaluation("P201", template_no_placeholders)
        print(f"Generated Summary (P201, no placeholders): '{summary_no_ph}'")
    except ValueError as e:
        print(f"Error: {e}")
