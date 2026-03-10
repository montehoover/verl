class Patient:
    def __init__(self, patient_id, name, age):
        self.patient_id = patient_id
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Patient(patient_id='{self.patient_id}', name='{self.name}', age={self.age})"

PATIENTS = {
    "101": Patient("101", "John Doe", 30),
    "102": Patient("102", "Jane Smith", 45),
    "103": Patient("103", "Alice Brown", 25),
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient object from the predefined dictionary by patient ID.

    Args:
        patient_id: The ID of the patient to retrieve.

    Returns:
        The Patient object corresponding to the given ID.

    Raises:
        ValueError: If the patient ID does not exist in the dictionary.
    """
    patient = PATIENTS.get(patient_id)
    if patient is None:
        raise ValueError(f"Patient with ID '{patient_id}' not found.")
    return patient

if __name__ == '__main__':
    # Example usage:
    try:
        patient1 = get_patient_by_id("101")
        print(f"Found patient: {patient1}")

        patient2 = get_patient_by_id("102")
        print(f"Found patient: {patient2}")

        # Example of a non-existent patient
        patient3 = get_patient_by_id("999")
        print(f"Found patient: {patient3}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example demonstrating the error for a non-existent ID
    try:
        get_patient_by_id("205")
    except ValueError as e:
        print(f"Caught expected error: {e}")
