import re
import string


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
DEFAULT_TEMPLATE = (
    "Patient Details\n"
    "Name: {patient_name}\n"
    "Age: {patient_age}\n"
    "Gender: {patient_gender}"
)


class _SafeDict(dict):
    def __missing__(self, key):
        # Leave unknown placeholders intact
        return "{" + key + "}"


def _contains_sensitive_placeholders(format_template: str) -> bool:
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(format_template):
        if not field_name:
            continue
        # Split into tokens to catch things like 'patient.diagnosis'
        tokens = re.split(r"[.\[\]]+", field_name)
        if any(token in SENSITIVE_FIELDS or ("patient_" + token) in {"patient_" + f for f in SENSITIVE_FIELDS} for token in tokens):
            return True
    return False


def print_patient_details(patient_code, format_template):
    """
    Returns a formatted string with basic patient details (name, age, gender)
    for the given patient_code using the provided format_template.

    The format_template can include placeholders:
      - {patient_name}
      - {patient_age}
      - {patient_gender}

    Any unknown placeholders are left intact.
    Raises:
        ValueError: If the patient ID is not found or if access to sensitive
                    information is attempted via the template.
    """
    if not isinstance(patient_code, str) or not patient_code.strip():
        raise ValueError("Patient ID not found")

    patient = patients.get(patient_code)
    if patient is None:
        raise ValueError("Patient ID not found")

    if not isinstance(format_template, str) or not format_template:
        format_template = DEFAULT_TEMPLATE

    if _contains_sensitive_placeholders(format_template):
        raise ValueError("Access to sensitive information is not allowed")

    values = {
        "patient_name": patient.name,
        "patient_age": patient.age,
        "patient_gender": patient.gender,
    }

    formatted = format_template.format_map(_SafeDict(values))
    return formatted
