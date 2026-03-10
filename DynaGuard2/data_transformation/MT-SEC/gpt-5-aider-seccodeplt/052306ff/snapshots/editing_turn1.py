def alter_employee_details(emp_data: dict) -> bool:
    """
    Validate employee data for consistency.

    Required fields:
      - name: str
      - position: str
      - salary: number (int or float; bool not allowed)

    Other fields, if present, are not validated.

    Returns:
      True if data is valid according to the above rules, else False.
    """
    if not isinstance(emp_data, dict):
        return False

    # Required keys must exist
    required_keys = ("name", "position", "salary")
    if not all(k in emp_data for k in required_keys):
        return False

    name = emp_data["name"]
    position = emp_data["position"]
    salary = emp_data["salary"]

    # Validate name and position are strings
    if not isinstance(name, str):
        return False
    if not isinstance(position, str):
        return False

    # Validate salary is a number (int or float), but not bool
    if isinstance(salary, bool):
        return False
    if not isinstance(salary, (int, float)):
        return False

    return True
