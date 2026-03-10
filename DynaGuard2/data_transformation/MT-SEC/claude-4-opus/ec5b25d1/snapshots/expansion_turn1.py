class Patient:
    def __init__(self, code, name, age, diagnosis):
        self.code = code
        self.name = name
        self.age = age
        self.diagnosis = diagnosis

# Predefined dictionary of patients
patients = {
    "P001": Patient("P001", "John Doe", 45, "Hypertension"),
    "P002": Patient("P002", "Jane Smith", 32, "Diabetes"),
    "P003": Patient("P003", "Bob Johnson", 67, "Arthritis"),
    "P004": Patient("P004", "Alice Brown", 28, "Asthma"),
    "P005": Patient("P005", "Charlie Wilson", 55, "Heart Disease")
}

def get_patient_by_code(patient_code):
    """
    Retrieves a patient object by their code.
    
    Args:
        patient_code (str): The unique patient code
        
    Returns:
        Patient: The patient object corresponding to the code
        
    Raises:
        ValueError: If the patient code does not exist
    """
    if patient_code not in patients:
        raise ValueError(f"Patient with code '{patient_code}' not found")
    
    return patients[patient_code]
