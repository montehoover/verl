class Patient:
    def __init__(self, patient_id, name, age, medical_record_number):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.medical_record_number = medical_record_number
    
    def __repr__(self):
        return f"Patient(id={self.patient_id}, name='{self.name}', age={self.age}, mrn='{self.medical_record_number}')"


# Predefined dictionary of patients
patients_db = {
    "P001": Patient("P001", "John Doe", 45, "MRN-2023-001"),
    "P002": Patient("P002", "Jane Smith", 32, "MRN-2023-002"),
    "P003": Patient("P003", "Robert Johnson", 67, "MRN-2023-003"),
    "P004": Patient("P004", "Maria Garcia", 28, "MRN-2023-004"),
    "P005": Patient("P005", "William Brown", 55, "MRN-2023-005")
}


def get_patient_by_id(patient_id):
    """
    Retrieve patient details by their ID.
    
    Args:
        patient_id (str): The unique identifier of the patient
        
    Returns:
        Patient: The Patient object corresponding to the given ID
        
    Raises:
        ValueError: If the patient ID is not found in the database
    """
    if patient_id in patients_db:
        return patients_db[patient_id]
    else:
        raise ValueError(f"Patient with ID '{patient_id}' not found")
