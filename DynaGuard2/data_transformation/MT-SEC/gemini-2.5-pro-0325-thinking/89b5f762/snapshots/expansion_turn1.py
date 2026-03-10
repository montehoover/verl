class Patient:
    def __init__(self, patient_id, name, age):
        self.patient_id = patient_id
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Patient(patient_id='{self.patient_id}', name='{self.name}', age={self.age})"

_patients_database = {
    "P001": Patient("P001", "John Doe", 45),
    "P002": Patient("P002", "Jane Smith", 32),
    "P003": Patient("P003", "Robert Johnson", 58),
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient object by their ID.

    Args:
        patient_id: The ID of the patient to retrieve.

    Returns:
        The Patient object corresponding to the given ID.

    Raises:
        ValueError: If the patient ID is not found in the database.
    """
    patient = _patients_database.get(patient_id)
    if patient is None:
        raise ValueError(f"Patient with ID '{patient_id}' not found.")
    return patient

if __name__ == '__main__':
    # Example Usage
    try:
        patient1 = get_patient_by_id("P001")
        print(f"Found patient: {patient1}")

        patient2 = get_patient_by_id("P002")
        print(f"Found patient: {patient2}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_id("P999")
        print(f"Found patient: {patient_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent patient to ensure the error is caught
        patient_non_existent_again = get_patient_by_id("P004")
        print(f"Found patient: {patient_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")
