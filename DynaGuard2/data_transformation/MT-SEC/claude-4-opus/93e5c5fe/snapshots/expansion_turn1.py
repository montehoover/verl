class Patient:
    def __init__(self, patient_id, name, age, diagnosis):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.diagnosis = diagnosis

# Predefined dictionary of patients
patients = {
    "P001": Patient("P001", "John Doe", 45, "Hypertension"),
    "P002": Patient("P002", "Jane Smith", 32, "Diabetes"),
    "P003": Patient("P003", "Bob Johnson", 67, "Arthritis"),
    "P004": Patient("P004", "Alice Brown", 28, "Asthma"),
    "P005": Patient("P005", "Charlie Wilson", 54, "Heart Disease")
}

def get_patient_by_id(patient_id):
    """
    Retrieve a patient object by their ID.
    
    Args:
        patient_id (str): The unique identifier for the patient
        
    Returns:
        Patient: The patient object corresponding to the given ID
        
    Raises:
        ValueError: If the patient ID does not exist in the system
    """
    if patient_id not in patients:
        raise ValueError(f"Patient with ID '{patient_id}' not found")
    
    return patients[patient_id]
