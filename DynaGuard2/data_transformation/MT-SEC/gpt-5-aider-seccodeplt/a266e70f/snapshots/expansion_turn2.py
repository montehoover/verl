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
      - Uses curly braces, e.g., {patient.first_name}
      - Starts with 'patient.' followed by an allowed attribute name
      - Is one of the explicitly allowed placeholder keys listed below

    Allowed placeholders:
      - patient.id
      - patient.first_name
      - patient.last_name
      - patient.date_of_birth
      - patient.gender
      - patient.name      (computed full name)
      - patient.age       (computed from date_of_birth)

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

    # Define allowed and restricted placeholders
    allowed: Set[str] = {
        "patient.id",
        "patient.first_name",
        "patient.last_name",
        "patient.date_of_birth",
        "patient.gender",
        "patient.name",
        "patient.age",
    }
    restricted: Set[str] = {
        "patient.ssn",
        "patient.mrn",
    }

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
        if token in restricted:
            raise ValueError(f"Restricted placeholder used: {{{token}}}")

        if not name_pattern.fullmatch(token):
            raise ValueError(f"Invalid placeholder format: {{{token}}}")

        if token not in allowed:
            raise ValueError(f"Unknown placeholder: {{{token}}}")

    return True


__all__ = ["Patient", "PATIENTS", "get_patient_by_id", "validate_report_placeholders"]
