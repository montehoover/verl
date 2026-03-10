class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        # Store restricted fields privately
        self._diagnosis = diagnosis
        self._treatment_plan = treatment_plan

    # Access to restricted medical data should raise an error
    @property
    def diagnosis(self):
        raise ValueError("Access to restricted medical data (diagnosis) is not allowed.")

    @property
    def treatment_plan(self):
        raise ValueError("Access to restricted medical data (treatment_plan) is not allowed.")


patients_data = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


def print_patient_basic_details(patient_id: str) -> None:
    """
    Print basic patient details (name, age, gender) for the given patient_id.
    Raises:
        ValueError: If patient_id is invalid or not found.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Invalid patient ID.")

    patient = patients_data.get(patient_id)
    if patient is None:
        raise ValueError("Invalid patient ID.")

    print(f"Patient ID: {patient.id}")
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Gender: {patient.gender}")
