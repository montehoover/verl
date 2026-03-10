from dataclasses import dataclass

@dataclass
class Patient:
    """Represents a patient."""
    patient_id: str
    name: str
    age: int
    condition: str

# Predefined dictionary of patients
# In a real application, this data would likely come from a database or external service.
_patients_database = {
    "P001": Patient(patient_id="P001", name="John Doe", age=45, condition="Hypertension"),
    "P002": Patient(patient_id="P002", name="Jane Smith", age=32, condition="Diabetes"),
    "P003": Patient(patient_id="P003", name="Alice Brown", age=58, condition="Arthritis"),
    "P004": Patient(patient_id="P004", name="Bob Johnson", age=67, condition="Asthma"),
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient's details by their ID.

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
    # Example usage:
    try:
        patient1 = get_patient_by_id("P001")
        print(f"Found patient: {patient1.name}, Age: {patient1.age}, Condition: {patient1.condition}")

        patient2 = get_patient_by_id("P003")
        print(f"Found patient: {patient2.name}, Age: {patient2.age}, Condition: {patient2.condition}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_id("P999")
        print(patient_non_existent) # This line won't be reached
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent patient
        patient_non_existent_2 = get_patient_by_id("P005")
        print(patient_non_existent_2) # This line won't be reached
    except ValueError as e:
        print(f"Error: {e}")
