import re
from typing import Dict, Set


class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


patients_data: Dict[str, Patient] = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

# Configuration and compiled regex used by the pure template filler
ALLOWED_FIELDS: Set[str] = {"name", "age", "gender"}
RESTRICTED_FIELDS: Set[str] = {"diagnosis", "treatment_plan"}
_PLACEHOLDER_PATTERN = re.compile(r'(?<!\{)\{patient\.(\w+)\}(?!\})')


def get_patient_by_id(patient_id: str, store: Dict[str, Patient]) -> Patient:
    """
    Pure function to fetch a Patient from a provided store by ID.
    Raises ValueError if the ID is invalid (not found).
    """
    patient = store.get(patient_id)
    if patient is None:
        raise ValueError("Invalid patient ID")
    return patient


def fill_patient_template(
    patient: Patient,
    template: str,
    allowed_fields: Set[str] = ALLOWED_FIELDS,
    restricted_fields: Set[str] = RESTRICTED_FIELDS,
    pattern=_PLACEHOLDER_PATTERN,
) -> str:
    """
    Pure function that fills a template using patient data.
    Raises ValueError if restricted fields are accessed.
    Unknown fields are left unchanged.
    """
    def replace(match: re.Match) -> str:
        field = match.group(1)

        if field in restricted_fields:
            raise ValueError("Attempt to access restricted medical data")

        if field in allowed_fields:
            return str(getattr(patient, field))

        # Unknown attribute: leave the placeholder unchanged
        return match.group(0)

    return pattern.sub(replace, template)


def generate_patient_summary(patient_id: str, report_template: str) -> str:
    """
    Generate a patient summary by filling in placeholders in the provided template.

    Args:
        patient_id: The unique identifier of the patient within the system.
        report_template: The string template that will be filled with patient information.
                         Placeholders should be of the form {patient.<field>}.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid or if the template attempts to access
                    restricted medical data.
    """
    patient = get_patient_by_id(patient_id, patients_data)
    return fill_patient_template(patient, report_template)
