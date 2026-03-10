class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


# Example in-memory datastore
patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

SENSITIVE_FIELDS = {"diagnosis", "treatment_plan"}


def print_patient_details(patient_id):
    """
    Print basic patient details (name, age, gender) for the given patient_id.

    Raises:
        ValueError: If the patient ID is not found, or if access to sensitive fields is attempted.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Patient ID not found")

    try:
        patient = patients[patient_id]
    except KeyError:
        raise ValueError("Patient ID not found")

    # Only allow these fields to be printed
    fields_to_print = ("name", "age", "gender")

    # Defensive check: prevent any accidental/sensitive field access
    if any(field in SENSITIVE_FIELDS for field in fields_to_print):
        raise ValueError("Access to sensitive patient information is not allowed")

    # Print formatted basic details
    print(f"Name: {getattr(patient, 'name')}")
    print(f"Age: {getattr(patient, 'age')}")
    print(f"Gender: {getattr(patient, 'gender')}")
