class Patient:
    def __init__(self, patient_id, name, age, medical_record_number):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.medical_record_number = medical_record_number

# Predefined dictionary of patients
patients = {
    "P001": Patient("P001", "John Doe", 45, "MRN12345"),
    "P002": Patient("P002", "Jane Smith", 32, "MRN12346"),
    "P003": Patient("P003", "Robert Johnson", 58, "MRN12347"),
    "P004": Patient("P004", "Maria Garcia", 27, "MRN12348"),
    "P005": Patient("P005", "David Brown", 63, "MRN12349")
}

def get_patient_by_id(patient_id):
    """
    Retrieve patient details using a patient ID.
    
    Args:
        patient_id (str): The patient ID to look up
        
    Returns:
        Patient: The corresponding Patient object
        
    Raises:
        ValueError: If the patient ID is not found
    """
    if patient_id in patients:
        return patients[patient_id]
    else:
        raise ValueError(f"Patient with ID '{patient_id}' not found")
