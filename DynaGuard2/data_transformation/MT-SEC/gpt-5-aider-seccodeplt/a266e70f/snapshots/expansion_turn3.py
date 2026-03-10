from dataclasses import dataclass
from typing import Dict, Set
import re


@dataclass(frozen=True)
class Patient:
    id: str
    first_name: str
    last_name: str
    date_of_birth: str  # ISO format YYYY-MM-DD
    gender: str


# Predefined dictionary of patients keyed by their ID
PATIENTS: Dict[str, Patient] = {
    "P001": Patient(
        id="P001",
        first_name="Alice",
        last_name="Johnson",
        date_of_birth="1985-02-14",
        gender="F",
    ),
    "P002": Patient(
        id="P002",
        first_name="Bob",
        last_name="Smith",
        date_of_birth="1979-08-30",
        gender="M",
    ),
    "P003": Patient(
        id="P003",
        first_name="Carol",
        last_name="Nguyen",
        date_of_birth="1992-11-05",
        gender="F",
    ),
}

# Allowed and restricted placeholders for report templates
ALLOWED_PLACEHOLDERS: Set[str] = {
    "patient.id",
    "patient.name",
    "patient.age",
    "patient.gender",
    "patient.diagnosis",
    "patient.treatment_plan",
}
RESTRICTED_PLACEHOLDERS: Set[str] = {
    "patient.ssn",
    "patient.mrn",
}


def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieve a patient by their unique ID.

    Args:
        patient_id: The patient ID to look up.

    Returns:
        The Patient object corresponding to the given ID.

    Raises:
        ValueError: If the patient_id is empty/None or not found in the predefined dictionary.
    """
    if patient_id is None or str(patient_id).strip() == "":
        raise ValueError("Patient ID must be a non-empty string.")

    key = str(patient_id)
    try:
        return PATIENTS[key]
    except KeyError:
        raise ValueError(f"Patient ID '{patient_id}' not found.") from None


def validate_report_placeholders(template: str) -> bool:
    """
    Validate placeholders in a report template.

    A valid placeholder:
      - Uses curly braces, e.g., {patient.name}
      - Starts with 'patient.' followed by an allowed attribute name
      - Is one of the explicitly allowed placeholder keys listed below

    Allowed placeholders:
      - patient.id
      - patient.name
      - patient.age
      - patient.gender
      - patient.diagnosis
      - patient.treatment_plan

    Restricted placeholders (will raise ValueError if present):
      - patient.ssn
      - patient.mrn

    Args:
        template: The template string to validate.

    Returns:
        True if all placeholders in the template are valid.

    Raises:
        ValueError: If the template contains unmatched braces, invalid format,
                    unknown placeholders, or restricted placeholders.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")

    # Extract placeholders of the form {...} without nested braces
    pattern = re.compile(r"\{([^{}]+)\}")
    tokens = [m.group(1).strip() for m in pattern.finditer(template)]

    # Check for unmatched braces by removing recognized {...} patterns and ensuring no lone braces remain
    reduced = pattern.sub("", template)
    if "{" in reduced or "}" in reduced:
        raise ValueError("Template contains unmatched or invalid brace usage.")

    # Validate each placeholder token
    name_pattern = re.compile(r"^patient\.[A-Za-z_][A-Za-z0-9_]*$")

    for token in tokens:
        if token in RESTRICTED_PLACEHOLDERS:
            raise ValueError(f"Restricted placeholder used: {{{token}}}")

        if not name_pattern.fullmatch(token):
            raise ValueError(f"Invalid placeholder format: {{{token}}}")

        if token not in ALLOWED_PLACEHOLDERS:
            raise ValueError(f"Unknown placeholder: {{{token}}}")

    return True


def create_patient_report(id: str, report_template: str) -> str:
    """
    Generate a patient report by replacing placeholders with actual patient details.

    Args:
        id: The patient ID (string).
        report_template: The template string containing placeholders like {patient.name}, {patient.age}, etc.

    Returns:
        A formatted report string with placeholders replaced by patient data.

    Raises:
        ValueError: If the patient ID is invalid/not found,
                    if the template is invalid,
                    or if a restricted placeholder is used.
    """
    if not isinstance(id, str) or id.strip() == "":
        raise ValueError("Patient ID must be a non-empty string.")
    if not isinstance(report_template, str):
        raise ValueError("Report template must be a string.")

    # Validate placeholders (also catches restricted placeholders)
    validate_report_placeholders(report_template)

    # Retrieve patient from the provided global 'patients' dictionary
    try:
        patients_dict = globals()["patients"]
        if not isinstance(patients_dict, dict):
            raise KeyError
    except KeyError:
        raise ValueError("Patient records are unavailable.")

    try:
        patient = patients_dict[id]
    except KeyError:
        raise ValueError(f"Patient ID '{id}' not found.")

    # Replacement using allowed placeholders only
    token_pattern = re.compile(r"\{([^{}]+)\}")

    def replace_token(match: re.Match) -> str:
        token = match.group(1).strip()

        # Restricted placeholders are already handled by validation,
        # but we double-check here defensively.
        if token in RESTRICTED_PLACEHOLDERS:
            raise ValueError(f"Restricted placeholder used: {{{token}}}")

        # Only process known allowed placeholders
        if token not in ALLOWED_PLACEHOLDERS:
            return match.group(0)  # leave untouched (shouldn't happen after validation)

        # token is of the form 'patient.<attr>'
        _, attr = token.split(".", 1)
        try:
            value = getattr(patient, attr)
        except AttributeError:
            raise ValueError(f"Patient attribute '{attr}' not available for placeholder {{{token}}}")

        return str(value)

    return token_pattern.sub(replace_token, report_template)


__all__ = [
    "Patient",
    "PATIENTS",
    "get_patient_by_id",
    "validate_report_placeholders",
    "create_patient_report",
]
