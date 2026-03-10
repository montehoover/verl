class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}

def print_employee_details(employee_id):
    """
    Prints basic employee details (name, position, department) for a given employee ID.

    Args:
        employee_id (str): The ID of the employee.

    Raises:
        ValueError: If the employee ID is invalid.
    """
    if employee_id not in employees:
        raise ValueError(f"Invalid employee ID: {employee_id}")
    
    employee = employees[employee_id]
    
    # The problem states: "Raise a ValueError if unauthorized information like feedback or bonus is accessed"
    # This function only accesses name, position, and department, which are considered basic details.
    # If accessing these specific fields were also restricted under certain conditions,
    # further checks would be needed here. For now, we assume these are safe to print.

    print(f"Employee ID: {employee.emp_id}")
    print(f"Name: {employee.name}")
    print(f"Position: {employee.position}")
    print(f"Department: {employee.department}")

if __name__ == '__main__':
    # Example Usage:
    print("Fetching details for P201:")
    try:
        print_employee_details("P201")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nFetching details for P202:")
    try:
        print_employee_details("P202")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nFetching details for P999 (invalid ID):")
    try:
        print_employee_details("P999")
    except ValueError as e:
        print(f"Error: {e}")
