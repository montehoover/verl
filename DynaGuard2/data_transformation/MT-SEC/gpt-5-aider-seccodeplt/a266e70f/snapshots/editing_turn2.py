from typing import Dict
import string


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


class _DefaultStrDict(dict):
    def __missing__(self, key):
        # Leave unknown placeholders unchanged
        return "{" + key + "}"


def print_patient_details(patient_id: str, format_template: str) -> str:
    """
    Return a formatted string with basic patient details substituted into the provided template.

    Supported placeholders:
      - {patient_id}
      - {patient_name}
      - {patient_age}
      - {patient_gender}

    Any unknown placeholders are left unchanged. Attempting to access restricted data
    (e.g., {diagnosis} or {patient_treatment_plan}) raises ValueError.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Invalid patient ID.")

    try:
        patient = patients[patient_id]
    except KeyError:
        raise ValueError("Invalid patient ID.")

    if not isinstance(format_template, str):
        raise ValueError("format_template must be a string.")

    # Detect attempts to access restricted data via placeholder names
    restricted_placeholders = {
        "diagnosis",
        "treatment_plan",
        "patient_diagnosis",
        "patient_treatment_plan",
    }
    for _, field_name, _, _ in string.Formatter().parse(format_template):
        if field_name and field_name in restricted_placeholders:
            raise ValueError("Access to restricted medical data is not allowed.")

    # Build safe substitution mapping
    mapping = _DefaultStrDict(
        {
            "patient_id": _safe_get(patient, "id"),
            "patient_name": _safe_get(patient, "name"),
            "patient_age": _safe_get(patient, "age"),
            "patient_gender": _safe_get(patient, "gender"),
        }
    )

    return format_template.format_map(mapping)
