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

performances = {
    "W101": Performance(employees["W101"], 4.8, "Led successful project launches", 3000),
    "W102": Performance(employees["W102"], 4.0, "Developed key modules on time", 2500),
}

def create_employee_report(worker_id: str, template: str) -> str:
    """
    Compiles a report summarizing an employee's performance based on a template.

    Args:
        worker_id: The unique ID of the worker.
        template: The template string for generating the summary.
                  Placeholders like '{worker.name}', '{worker.position}', 
                  '{assessment.rating}' are supported.

    Returns:
        The formatted performance summary string.

    Raises:
        ValueError: If unauthorized information (e.g., feedback, bonus) is accessed,
                    or if the worker ID is invalid, or if the template contains
                    invalid placeholders.
    """
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")

    worker = employees[worker_id]
    
    if worker_id not in performances:
        # This case implies an inconsistency if an employee must have a performance record.
        # Based on the problem structure, we assume a performance record should exist.
        raise ValueError(f"Performance record not found for worker ID: {worker_id}")
        
    assessment = performances[worker_id]

    # Check for unauthorized placeholders in the template
    # Unauthorized fields are feedback and bonus from the assessment
    forbidden_placeholders = ["{assessment.feedback}", "{assessment.bonus}"]
    for placeholder in forbidden_placeholders:
        if placeholder in template:
            raise ValueError(
                f"Access to unauthorized information ('{placeholder.strip('{}')}') in template is forbidden."
            )

    try:
        # Populate the template using the worker and assessment objects
        # Allowed fields for worker: name, position, department, emp_id
        # Allowed fields for assessment: rating
        report = template.format(worker=worker, assessment=assessment)
    except (AttributeError, KeyError) as e:
        # This catches errors if the template tries to access attributes that don't exist
        # on the worker or assessment objects (e.g., {worker.age} or {assessment.salary})
        # or if the placeholder syntax is incorrect.
        raise ValueError(f"Invalid placeholder or attribute in template: {e}")
    
    return report

if __name__ == '__main__':
    # Example Usage as per the problem description
    try:
        report1 = create_employee_report(
            worker_id="W101",
            template="Name: {worker.name}, Position: {worker.position}, Rating: {assessment.rating}"
        )
        print(f"Report for W101: {report1}")
        # Expected: Name: Alice Johnson, Position: Project Manager, Rating: 4.8

        report2 = create_employee_report(
            worker_id="W102",
            template="Employee {worker.name} ({worker.emp_id}) from {worker.department} has a rating of {assessment.rating}."
        )
        print(f"Report for W102: {report2}")
        # Expected: Employee Bob Lee (W102) from IT has a rating of 4.0.

        # Example of unauthorized access
        try:
            create_employee_report(
                worker_id="W101",
                template="Name: {worker.name}, Bonus: {assessment.bonus}"
            )
        except ValueError as e:
            print(f"Error (unauthorized access): {e}")
            # Expected: Error (unauthorized access): Access to unauthorized information ('assessment.bonus') in template is forbidden.

        # Example of invalid worker ID
        try:
            create_employee_report(
                worker_id="W999",
                template="Name: {worker.name}"
            )
        except ValueError as e:
            print(f"Error (invalid ID): {e}")
            # Expected: Error (invalid ID): Invalid worker ID: W999

        # Example of invalid placeholder attribute
        try:
            create_employee_report(
                worker_id="W101",
                template="Name: {worker.nonexistent_field}"
            )
        except ValueError as e:
            print(f"Error (invalid placeholder): {e}")
            # Expected: Error (invalid placeholder): Invalid placeholder or attribute in template: 'Employee' object has no attribute 'nonexistent_field'

    except ValueError as e:
        print(f"An unexpected error occurred: {e}")
