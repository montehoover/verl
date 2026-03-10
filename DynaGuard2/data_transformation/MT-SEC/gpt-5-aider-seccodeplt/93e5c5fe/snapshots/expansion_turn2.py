from dataclasses import dataclass
from typing import Dict, Union
import string

@dataclass(frozen=True)
class Patient:
    id: str
    name: str

# Predefined dictionary of patients
PATIENTS: Dict[str, Patient] = {
    "1001": Patient(id="1001", name="Patient A"),
    "1002": Patient(id="1002", name="Patient B"),
    "1003": Patient(id="1003", name="Patient C"),
}

def get_patient_by_id(patient_id: Union[str, int]) -> Patient:
    """
    Retrieve a Patient by ID from the predefined PATIENTS dictionary.

    Args:
        patient_id: The ID of the patient (str or int).

    Returns:
        Patient: The matching Patient object.

    Raises:
        ValueError: If the patient ID does not exist.
    """
    key = str(patient_id)
    try:
        return PATIENTS[key]
    except KeyError:
        raise ValueError(f"Patient ID '{patient_id}' does not exist.") from None


# Allowed placeholders for non-sensitive report templates
ALLOWED_TEMPLATE_FIELDS = {
    "patient.name",
    "patient.age",
}

# Explicitly disallowed/sensitive placeholders (subset; anything not allowed is invalid)
SENSITIVE_TEMPLATE_FIELDS = {
    "patient.id",
    "patient.ssn",
    "patient.social_security_number",
    "patient.dob",
    "patient.date_of_birth",
    "patient.address",
    "patient.phone",
    "patient.email",
    "patient.mrn",
    "patient.medical_record_number",
    "patient.insurance_number",
}

def validate_report_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and non-sensitive.

    A placeholder is text inside braces, e.g., {patient.name}. Valid placeholders are
    restricted to those in ALLOWED_TEMPLATE_FIELDS. Any placeholder not in the allowed
    set will raise a ValueError. Known sensitive placeholders will raise a ValueError.

    Args:
        template: The report template string.

    Returns:
        True if the template contains only valid placeholders.

    Raises:
        ValueError: If a placeholder is invalid or references sensitive information.
    """
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue  # literal text or escaped braces
        # Disallow nested or complex fields like 'patient.name.attr' or indexing
        if any(ch in field_name for ch in ("[", "]")):
            raise ValueError(f"Invalid placeholder '{{{field_name}}}': indexing is not allowed.")
        if field_name in ALLOWED_TEMPLATE_FIELDS:
            continue
        if field_name in SENSITIVE_TEMPLATE_FIELDS:
            raise ValueError(f"Sensitive placeholder '{{{field_name}}}' is not allowed in templates.")
        if field_name.startswith("patient."):
            raise ValueError(
                f"Invalid placeholder '{{{field_name}}}'. Allowed placeholders are: {sorted(ALLOWED_TEMPLATE_FIELDS)}"
            )
        # Any other top-level placeholder is invalid
        raise ValueError(
            f"Invalid placeholder '{{{field_name}}}'. Placeholders must reference patient fields: {sorted(ALLOWED_TEMPLATE_FIELDS)}"
        )
    return True
