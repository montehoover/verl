class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, worker, rating, feedback, bonus):
        self.worker = worker
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "W101": Employee("W101", "Alice Johnson", "Project Manager", "Operations"),
    "W102": Employee("W102", "Bob Lee", "Software Developer", "IT"),
}

import re

performances = {
    "W101": Performance(employees["W101"], 4.8, "Led successful project launches", 3000),
    "W102": Performance(employees["W102"], 4.0, "Developed key modules on time", 2500),
}

def create_employee_report(worker_id, template):
    """
    Creates a formatted employee report string based on a template.
    The template can use placeholders like {worker.name} or {assessment.rating}.

    Args:
        worker_id (str): The ID of the employee.
        template (str): A string template with placeholders.
                        Supported objects: 'worker', 'assessment'.
                        Supported worker attributes: 'emp_id', 'name', 'position', 'department'.
                        Supported assessment attributes: 'rating', 'feedback', 'bonus'.

    Returns:
        str: The formatted performance summary string.

    Raises:
        ValueError: If the worker_id is invalid, or if performance data is missing.
                    If the template attempts to access unauthorized objects or attributes.
    """
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    employee_obj = employees[worker_id]

    if worker_id not in performances:
        raise ValueError(f"Performance data not found for worker ID: {worker_id}")
    performance_obj = performances[worker_id]

    allowed_worker_attrs = {"emp_id", "name", "position", "department"}
    allowed_assessment_attrs = {"rating", "feedback", "bonus"}

    def replacer(match):
        placeholder_content = match.group(1)  # Content inside {} e.g., "worker.name"
        parts = placeholder_content.split('.', 1)

        if len(parts) == 2:
            obj_key, attr_key = parts[0], parts[1]

            if obj_key == "worker":
                if attr_key in allowed_worker_attrs:
                    return str(getattr(employee_obj, attr_key))
                else:
                    raise ValueError(f"Access to unauthorized attribute '{attr_key}' for 'worker' in placeholder '{{{placeholder_content}}}' is not allowed.")
            elif obj_key == "assessment":
                if attr_key in allowed_assessment_attrs:
                    return str(getattr(performance_obj, attr_key))
                else:
                    raise ValueError(f"Access to unauthorized attribute '{attr_key}' for 'assessment' in placeholder '{{{placeholder_content}}}' is not allowed.")
            else:
                raise ValueError(f"Unknown object '{obj_key}' in placeholder '{{{placeholder_content}}}'.")
        else:
            # Placeholder doesn't match "object.attribute" format, leave it as is.
            return match.group(0)

    formatted_string = re.sub(r"\{(.*?)\}", replacer, template)
    return formatted_string

if __name__ == '__main__':
    # Example usage:
    template1 = "Employee Report:\nID: {worker.emp_id}\nName: {worker.name}\nPosition: {worker.position}\nDepartment: {worker.department}"
    template2 = "Performance Summary for {worker.name} ({worker.emp_id}):\nRole: {worker.position}\nRating: {assessment.rating}\nFeedback: {assessment.feedback}\nBonus: ${assessment.bonus}\nSimple Placeholder: {unformatted}"

    print("Generating report for W101 (Template 1):")
    try:
        report1 = create_employee_report("W101", template1)
        print(report1)
    except ValueError as e:
        print(e)

    print("\nGenerating report for W102 (Template 2):")
    try:
        report2 = create_employee_report("W102", template2)
        print(report2)
    except ValueError as e:
        print(e)

    print("\nGenerating report for W103 (invalid worker):")
    try:
        report3 = create_employee_report("W103", template1)
        print(report3)
    except ValueError as e:
        print(e)
    
    # Add a worker without performance data to test that case
    employees["W104"] = Employee("W104", "Charlie Brown", "Intern", "Peanuts")
    print("\nGenerating report for W104 (worker with no performance data):")
    try:
        report4 = create_employee_report("W104", template1) # Template 1 only uses worker fields
        print(report4) # This should work if template only uses worker fields and performance data is not strictly needed for them
        report_perf = create_employee_report("W104", template2) # Template 2 uses assessment fields
        print(report_perf)
    except ValueError as e:
        print(e) # Expected: Performance data not found for worker ID: W104

    del employees["W104"] # Clean up

    print("\nAttempting to generate report with unauthorized worker attribute:")
    template_unauth_worker_attr = "Name: {worker.name}, Secret: {worker.secret_key}"
    try:
        report_unauth_w = create_employee_report("W101", template_unauth_worker_attr)
        print(report_unauth_w)
    except ValueError as e:
        print(e)

    print("\nAttempting to generate report with unauthorized assessment attribute:")
    template_unauth_assessment_attr = "Rating: {assessment.rating}, Secret: {assessment.secret_info}"
    try:
        report_unauth_a = create_employee_report("W101", template_unauth_assessment_attr)
        print(report_unauth_a)
    except ValueError as e:
        print(e)

    print("\nAttempting to generate report with unknown object:")
    template_unknown_obj = "Name: {worker.name}, Data: {unknown_source.data}"
    try:
        report_unknown_obj = create_employee_report("W101", template_unknown_obj)
        print(report_unknown_obj)
    except ValueError as e:
        print(e)
