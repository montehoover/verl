from typing import Dict

class Performance:
    """Represents an employee's performance data."""
    def __init__(self, rating: float, review: str):
        self.rating = rating
        self.review = review

    def __repr__(self) -> str:
        return f"Performance(rating={self.rating}, review='{self.review}')"

# Predefined dictionary of employee performances
# In a real application, this data might come from a database or an external service.
_employee_performances: Dict[int, Performance] = {
    101: Performance(rating=4.5, review="Exceeds expectations in all areas."),
    102: Performance(rating=3.0, review="Meets expectations, room for growth in project management."),
    103: Performance(rating=4.0, review="Strong performer, consistently delivers high-quality work."),
    104: Performance(rating=2.5, review="Needs improvement in communication skills."),
}

def get_performance_data(employee_id: int) -> Performance:
    """
    Fetches the performance data for a given employee ID.

    Args:
        employee_id: The ID of the employee.

    Returns:
        The Performance object for the employee.

    Raises:
        ValueError: If the employee ID does not exist in the performance records.
    """
    if employee_id in _employee_performances:
        return _employee_performances[employee_id]
    else:
        raise ValueError(f"No performance data found for employee ID: {employee_id}")

if __name__ == '__main__':
    # Example usage:
    try:
        perf_101 = get_performance_data(101)
        print(f"Employee 101 Performance: {perf_101}")

        perf_102 = get_performance_data(102)
        print(f"Employee 102 Performance: {perf_102}")

        # Example of an employee ID that does not exist
        perf_999 = get_performance_data(999)
        print(f"Employee 999 Performance: {perf_999}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent ID
        perf_105 = get_performance_data(105)
        print(f"Employee 105 Performance: {perf_105}")
    except ValueError as e:
        print(f"Error: {e}")
