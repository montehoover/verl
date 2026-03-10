from dataclasses import dataclass
from typing import Dict, List
from string import Formatter


@dataclass(frozen=True)
class Patient:
    code: str
    first_name: str
    last_name: str
    date_of_birth: str  # ISO format YYYY-MM-DD
    conditions: List[str]


# Predefined dictionary of patients keyed by patient code
PATIENTS: Dict[str, Patient] = {
    "P001": Patient(
        code="P001",
        first_name="Alex",
        last_name="Doe",
        date_of_birth="1980-01-01",
        conditions=["Hypertension"],
    ),
    "P002": Patient(
        code="P002",
        first_name="Riley",
        last_name="Smith",
        date_of_birth="1975-05-15",
        conditions=["Type 2 Diabetes"],
    ),
    "P003": Patient(
        code="P003",
        first_name="Jordan",
        last_name="Lee",
        date_of_birth="1990-09-30",
        conditions=["Asthma"],
    ),
}


def get_patient_by_code(patient_code: str) -> Patient:
    """
    Return the Patient corresponding to the given patient_code.

    Raises:
        ValueError: If the patient_code does not exist in the predefined dictionary.
    """
    try:
        return PATIENTS[patient_code]
    except KeyError:
        raise ValueError(f"Patient code not found: {patient_code}")


# Allowed non-sensitive placeholders in report templates
ALLOWED_TEMPLATE_PLACEHOLDERS = {
    "patient.name",
    "patient.age",
}


def validate_report_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed non-sensitive placeholders.

    Allowed placeholders:
      - {patient.name}
      - {patient.age}

    Returns:
        bool: True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or sensitive placeholder is detected.
    """
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if not field_name:
            # No placeholder in this segment (or escaped braces), skip
            continue
        if field_name not in ALLOWED_TEMPLATE_PLACEHOLDERS:
            raise ValueError(f"Invalid or sensitive placeholder: {{{field_name}}}")
    return True
