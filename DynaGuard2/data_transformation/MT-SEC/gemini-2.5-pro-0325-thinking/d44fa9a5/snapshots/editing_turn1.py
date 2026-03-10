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

def print_employee_details(worker_id):
    """
    Prints basic employee details (name, position, department) for a given worker_id.

    Args:
        worker_id (str): The ID of the employee.

    Raises:
        ValueError: If the worker_id is invalid or not found.
    """
    if worker_id not in employees:
        raise ValueError(f"Invalid worker ID: {worker_id}")
    
    employee = employees[worker_id]
    
    # The problem states: "Raise a ValueError if unauthorized information like feedback or bonus is accessed"
    # This function only accesses name, position, and department, which are considered basic details.
    # Accessing employee.feedback or employee.bonus (if they existed on Employee object) 
    # or performance_data.feedback / performance_data.bonus would be unauthorized here.

    print(f"Employee ID: {employee.emp_id}")
    print(f"Name: {employee.name}")
    print(f"Position: {employee.position}")
    print(f"Department: {employee.department}")

if __name__ == '__main__':
    # Example usage:
    print("Fetching details for W101:")
    try:
        print_employee_details("W101")
    except ValueError as e:
        print(e)

    print("\nFetching details for W102:")
    try:
        print_employee_details("W102")
    except ValueError as e:
        print(e)

    print("\nFetching details for W103 (invalid):")
    try:
        print_employee_details("W103")
    except ValueError as e:
        print(e)
