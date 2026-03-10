import re
from typing import Dict


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


employees: Dict[str, Employee] = {
    "W101": Employee("W101", "Alice Johnson", "Project Manager", "Operations"),
    "W102": Employee("W102", "Bob Lee", "Software Developer", "IT"),
}

performances: Dict[str, Performance] = {
    "W101": Performance(employees["W101"], 4.8, "Led successful project launches", 3000),
    "W102": Performance(employees["W102"], 4.0, "Developed key modules on time", 2500),
}


def create_employee_report(worker_id: str, template: str) -> str:
    """
    Create a formatted performance summary for a worker based on a template.

    Args:
        worker_id: The unique ID of the worker.
        template: The template string containing placeholders like
                  '{worker.name}', '{worker.position}', '{assessment.rating}'.

    Returns:
        The formatted performance summary string.

    Raises:
        ValueError: If the worker ID is invalid or if unauthorized information
                    (like feedback or bonus) is requested in the template.
    """
    # Validate worker ID and retrieve associated records
    worker = employees.get(worker_id)
    perf = performances.get(worker_id)
    if worker is None or perf is None:
        raise ValueError("Invalid worker ID")

    # Find all placeholders of the form {something}
    pattern = re.compile(r"\{([^{}]+)\}")
    placeholders = set(pattern.findall(template))

    # Prepare replacements while enforcing authorization rules
    replacements: Dict[str, str] = {}
    for ph in placeholders:
        key = ph.strip()

        if key.startswith("worker."):
            attr = key.split(".", 1)[1]
            # Allow only known safe attributes of Employee
            if attr in {"name", "position", "department", "emp_id"}:
                replacements[key] = str(getattr(worker, attr))
            # Unknown worker attributes are left unchanged (no replacement)

        elif key.startswith("assessment."):
            attr = key.split(".", 1)[1]
            # Explicitly forbid sensitive attributes
            if attr in {"feedback", "bonus"}:
                raise ValueError("Unauthorized access to sensitive performance information")
            if attr == "rating":
                replacements[key] = str(perf.rating)
            # Unknown assessment attributes are left unchanged (no replacement)

        # Any other prefix is left unchanged

    # Apply replacements
    result = template
    for k, v in replacements.items():
        result = result.replace("{" + k + "}", v)

    return result
