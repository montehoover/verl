class Employee:
    def __init__(self, name, position):
        self.name = name
        self.position = position

    def __repr__(self):
        return f"Employee(name='{self.name}', position='{self.position}')"

_employees_database = {
    "101": Employee("Alice Wonderland", "Software Engineer"),
    "102": Employee("Bob The Builder", "Project Manager"),
    "103": Employee("Charlie Brown", "Data Scientist"),
}

def get_employee_by_id(worker_id: str) -> Employee:
    """
    Retrieves an employee object by their worker ID.

    Args:
        worker_id: The ID of the worker to retrieve.

    Returns:
        The Employee object corresponding to the worker ID.

    Raises:
        ValueError: If the worker ID does not exist in the database.
    """
    employee = _employees_database.get(worker_id)
    if employee is None:
        raise ValueError(f"Employee with ID '{worker_id}' not found.")
    return employee

if __name__ == '__main__':
    # Example usage:
    try:
        employee1 = get_employee_by_id("101")
        print(f"Found employee: {employee1}")

        employee2 = get_employee_by_id("102")
        print(f"Found employee: {employee2}")

        # Example of a non-existent ID
        employee_non_existent = get_employee_by_id("999")
        print(f"Found employee: {employee_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent ID
        employee_non_existent_again = get_employee_by_id("104")
        print(f"Found employee: {employee_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")
