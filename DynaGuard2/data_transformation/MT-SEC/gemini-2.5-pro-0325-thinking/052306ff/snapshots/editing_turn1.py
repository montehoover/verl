from typing import Dict, Any

def alter_employee_details(emp_data: Dict[str, Any]) -> bool:
    """
    Checks employee data for consistency.

    Args:
        emp_data: A dictionary containing employee data.
                  Expected keys include 'name', 'position', 'salary'.

    Returns:
        True if the data is valid, False otherwise.
    """
    if not isinstance(emp_data.get("name"), str):
        return False
    if not isinstance(emp_data.get("position"), str):
        return False
    salary = emp_data.get("salary")
    if not isinstance(salary, (int, float)):
        return False
    return True
