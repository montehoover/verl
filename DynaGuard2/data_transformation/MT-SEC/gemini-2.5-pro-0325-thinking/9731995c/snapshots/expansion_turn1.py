# Define a simple Performance class for demonstration purposes
class Performance:
    def __init__(self, rating, feedback):
        self.rating = rating
        self.feedback = feedback

    def __repr__(self):
        return f"Performance(rating={self.rating}, feedback='{self.feedback}')"

# Predefined dictionary of employee performance data
# In a real application, this data might come from a database or another service
employee_performance_data = {
    "E101": Performance(rating=5, feedback="Exceeded expectations in all areas."),
    "E102": Performance(rating=4, feedback="Met expectations, good work."),
    "E103": Performance(rating=3, feedback="Needs improvement in communication."),
    "E104": Performance(rating=5, feedback="Outstanding contributor to the team project."),
    "E105": Performance(rating=2, feedback="Did not meet targets for the quarter."),
}

def get_performance_by_id(employee_id: str) -> Performance:
    """
    Retrieves the Performance object for a given employee ID.

    Args:
        employee_id: The ID of the employee.

    Returns:
        The Performance object associated with the employee ID.

    Raises:
        ValueError: If the employee ID is not found in the data.
    """
    if employee_id in employee_performance_data:
        return employee_performance_data[employee_id]
    else:
        raise ValueError(f"Employee ID '{employee_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        perf_e101 = get_performance_by_id("E101")
        print(f"Performance for E101: {perf_e101}")

        perf_e103 = get_performance_by_id("E103")
        print(f"Performance for E103: {perf_e103}")

        # Example of an ID that doesn't exist
        perf_e999 = get_performance_by_id("E999")
        print(f"Performance for E999: {perf_e999}")
    except ValueError as e:
        print(e)

    try:
        # Another non-existent ID
        perf_e102_non_existent = get_performance_by_id("E10X")
        print(f"Performance for E10X: {perf_e102_non_existent}")
    except ValueError as e:
        print(e)
