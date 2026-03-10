from typing import Dict


class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


patients: Dict[str, Patient] = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

_RESTRICTED_ATTRS = {"diagnosis", "treatment_plan"}


def _safe_get(patient: Patient, attr: str):
    if attr in _RESTRICTED_ATTRS:
        raise ValueError("Access to restricted medical data is not allowed.")
    return getattr(patient, attr)


def print_patient_details(patient_id: str) -> None:
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Invalid patient ID.")

    try:
        patient = patients[patient_id]
    except KeyError:
        raise ValueError("Invalid patient ID.")

    name = _safe_get(patient, "name")
    age = _safe_get(patient, "age")
    gender = _safe_get(patient, "gender")

    print(f"Name: {name}\nAge: {age}\nGender: {gender}")
