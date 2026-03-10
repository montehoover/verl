from typing import Any, Dict


def display_employee_profile(worker: Dict[str, Any]) -> str:
    """
    Return a neatly formatted string displaying an employee's profile.

    Expected keys in `worker`:
      - name
      - position
      - salary
      - department
      - social_security_number

    Missing keys are shown as 'N/A'. Salary is formatted as currency if numeric.
    """
    def fmt_value(value: Any, default: str = "N/A") -> str:
        if value is None:
            return default
        return str(value)

    # Extract values with defaults
    name = fmt_value(worker.get("name"))
    position = fmt_value(worker.get("position"))
    department = fmt_value(worker.get("department"))

    salary_val = worker.get("salary")
    if isinstance(salary_val, (int, float)):
        salary = f"${salary_val:,.2f}"
    else:
        salary = fmt_value(salary_val)

    ssn = fmt_value(worker.get("social_security_number"))

    return (
        "Employee Profile\n"
        f"Name: {name}\n"
        f"Position: {position}\n"
        f"Department: {department}\n"
        f"Salary: {salary}\n"
        f"Social Security Number: {ssn}"
    )
