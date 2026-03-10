class Employee:
    def __init__(self, id, name, position):
        self.id = id
        self.name = name
        self.position = position

    def __repr__(self):
        return f"Employee(id={self.id}, name='{self.name}', position='{self.position}')"

# Predefined dictionary of employees
# In a real application, this data might come from a database or an external file.
_employees_data = {
    101: Employee(101, "Alice Wonderland", "Software Engineer"),
    102: Employee(102, "Bob The Builder", "Project Manager"),
    103: Employee(103, "Charlie Brown", "Data Scientist"),
    104: Employee(104, "Diana Prince", "UX Designer"),
}

def get_employee_by_id(employee_id: int) -> Employee:
    """
    Retrieves an employee object by their ID.

    Args:
        employee_id: The ID of the employee to retrieve.

    Returns:
        The Employee object corresponding to the given ID.

    Raises:
        ValueError: If an employee with the given ID does not exist.
    """
    employee = _employees_data.get(employee_id)
    if employee is None:
        raise ValueError(f"Employee with ID {employee_id} not found.")
    return employee

if __name__ == '__main__':
    # Example usage:
    try:
        employee = get_employee_by_id(101)
        print(f"Found employee: {employee}")

        employee_non_existent = get_employee_by_id(999)
        print(f"Found employee: {employee_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        employee_bob = get_employee_by_id(102)
        print(f"Found employee: {employee_bob}")
    except ValueError as e:
        print(f"Error: {e}")
