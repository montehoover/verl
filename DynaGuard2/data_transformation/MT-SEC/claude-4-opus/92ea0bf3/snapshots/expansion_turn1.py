class Performance:
    def __init__(self, employee_id, rating, goals_met, review_notes):
        self.employee_id = employee_id
        self.rating = rating
        self.goals_met = goals_met
        self.review_notes = review_notes

# Predefined dictionary of performances
performances = {
    "EMP001": Performance("EMP001", 4.5, True, "Excellent work on Q3 projects"),
    "EMP002": Performance("EMP002", 3.8, True, "Good performance, needs improvement in communication"),
    "EMP003": Performance("EMP003", 4.2, False, "Strong technical skills, missed deadline on one goal"),
    "EMP004": Performance("EMP004", 4.9, True, "Outstanding contributor, exceeded all expectations"),
}

def get_performance_data(employee_id):
    if employee_id not in performances:
        raise ValueError(f"Employee ID {employee_id} does not exist")
    return performances[employee_id]
