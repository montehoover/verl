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


def get_employee(worker_id: str) -> Employee:
    """
    Retrieve an Employee by worker_id.

    Raises:
        ValueError: If no employee exists for the given worker_id.
    """
    worker = employees.get(worker_id)
    if worker is None:
        raise ValueError("Invalid worker ID")
    return worker


def get_performance(worker_id: str) -> Performance:
    """
    Retrieve a Performance record by worker_id.

    Raises:
        ValueError: If no performance record exists for the given worker_id.
    """
    perf = performances.get(worker_id)
    if perf is None:
        raise ValueError("Invalid worker ID")
    return perf


def process_template(template: str, worker: Employee, perf: Performance) -> str:
    """
    Process the template string, replacing placeholders with values from
    the provided worker and performance objects.

    Raises:
        ValueError: If unauthorized information (feedback or bonus) is requested.
    """
    pattern = re.compile(r"\{([^{}]+)\}")
    placeholders = set(pattern.findall(template))

    replacements: Dict[str, str] = {}
    for ph in placeholders:
        key = ph.strip()

        if key.startswith("worker."):
            attr = key.split(".", 1)[1]
            if attr in {"name", "position", "department", "emp_id"}:
                replacements[key] = str(getattr(worker, attr))

        elif key.startswith("assessment."):
            attr = key.split(".", 1)[1]
            if attr in {"feedback", "bonus"}:
                raise ValueError("Unauthorized access to sensitive performance information")
            if attr == "rating":
                replacements[key] = str(perf.rating)

    result = template
    for k, v in replacements.items():
        result = result.replace("{" + k + "}", v)

    return result


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
    worker = get_employee(worker_id)
    perf = get_performance(worker_id)
    return process_template(template, worker, perf)
