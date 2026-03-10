class Performance:
    def __init__(self, employee_id, rating, review):
        self.employee_id = employee_id
        self.rating = rating
        self.review = review

# Predefined dictionary of performance data
performance_data = {
    101: Performance(101, 4.5, "Excellent work on project deliverables"),
    102: Performance(102, 3.8, "Good teamwork and communication skills"),
    103: Performance(103, 4.2, "Strong technical skills and problem-solving"),
    104: Performance(104, 3.5, "Meets expectations, room for improvement"),
    105: Performance(105, 4.8, "Outstanding leadership and innovation")
}

def get_performance_by_id(employee_id):
    """
    Retrieves the Performance object for a given employee ID.
    
    Args:
        employee_id: The ID of the employee
        
    Returns:
        Performance object associated with the employee ID
        
    Raises:
        ValueError: If the employee ID doesn't exist in the performance data
    """
    if employee_id not in performance_data:
        raise ValueError(f"Employee ID {employee_id} not found in performance data")
    
    return performance_data[employee_id]
