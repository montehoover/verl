class Patient:
    """Represents a patient."""
    def __init__(self, code, name, age):
        self.code = code
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Patient(code='{self.code}', name='{self.name}', age={self.age})"

_patients_database = {
    "P001": Patient("P001", "John Doe", 45),
    "P002": Patient("P002", "Jane Smith", 32),
    "P003": Patient("P003", "Alice Brown", 58),
}

def get_patient_by_code(patient_code: str) -> Patient:
    """
    Retrieves a patient object from the predefined dictionary using their code.

    Args:
        patient_code: The unique code of the patient.

    Returns:
        The Patient object corresponding to the given code.

    Raises:
        ValueError: If the patient code does not exist in the database.
    """
    patient = _patients_database.get(patient_code)
    if patient is None:
        raise ValueError(f"Patient with code '{patient_code}' not found.")
    return patient

if __name__ == '__main__':
    # Example usage:
    try:
        patient1 = get_patient_by_code("P001")
        print(f"Found patient: {patient1}")

        patient2 = get_patient_by_code("P002")
        print(f"Found patient: {patient2}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_code("P999")
        print(f"Found patient: {patient_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent patient to show the error again
        patient_non_existent_again = get_patient_by_code("P004")
        print(f"Found patient: {patient_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")
