class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

SENSITIVE_FIELDS = {"diagnosis", "treatment_plan"}


def print_patient_details(patient_code):
    """
    Prints basic patient details (name, age, gender) for the given patient_code.

    Raises:
        ValueError: If the patient ID is not found or if access to sensitive information is attempted.
    """
    if not isinstance(patient_code, str) or not patient_code.strip():
        raise ValueError("Patient ID not found")

    patient = patients.get(patient_code)
    if patient is None:
        raise ValueError("Patient ID not found")

    # Only allow non-sensitive fields
    requested_fields = ("name", "age", "gender")
    if any(field in SENSITIVE_FIELDS for field in requested_fields):
        raise ValueError("Access to sensitive information is not allowed")

    output = (
        f"Patient Details\n"
        f"Name: {patient.name}\n"
        f"Age: {patient.age}\n"
        f"Gender: {patient.gender}"
    )
    print(output)
