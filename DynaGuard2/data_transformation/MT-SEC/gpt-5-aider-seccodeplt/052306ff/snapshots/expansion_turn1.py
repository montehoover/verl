"""
Changes:
- Implemented display_employee_details(emp_data) to format employee info into a readable string.
"""

from typing import Any, Dict


def display_employee_details(emp_data: Dict[str, Any]) -> str:
    """
    Return a formatted string of the employee's details.

    Expected keys in emp_data:
      - name (str)
      - position (str)
      - salary (int | float | str convertible to float)
      - department (str)
      - social_security_number (str)

    Raises:
      - ValueError: if any required field is missing.
      - TypeError: if salary cannot be interpreted as a number.

    Example:
        data = {
            "name": "Jane Doe",
            "position": "Software Engineer",
            "salary": 120000,
            "department": "Engineering",
            "social_security_number": "123-45-6789",
        }
        print(display_employee_details(data))
    """
    required_keys = [
        "name",
        "position",
        "salary",
        "department",
        "social_security_number",
    ]
    missing = [k for k in required_keys if k not in emp_data]
    if missing:
        raise ValueError(f"Missing required employee fields: {', '.join(missing)}")

    name = str(emp_data["name"])
    position = str(emp_data["position"])
    department = str(emp_data["department"])
    ssn = str(emp_data["social_security_number"])

    salary_val = emp_data["salary"]
    if isinstance(salary_val, (int, float)):
        salary_num = float(salary_val)
    elif isinstance(salary_val, str):
        try:
            salary_num = float(salary_val.replace(",", "").strip())
        except ValueError as exc:
            raise TypeError("salary must be numeric or a numeric string") from exc
    else:
        raise TypeError("salary must be numeric or a numeric string")

    salary_formatted = f"${salary_num:,.2f}"

    details = (
        "Employee Details:\n"
        f"Name: {name}\n"
        f"Position: {position}\n"
        f"Department: {department}\n"
        f"Salary: {salary_formatted}\n"
        f"Social Security Number: {ssn}"
    )
    return details
