from dataclasses import dataclass
from typing import Dict, List, Set
import string


@dataclass(frozen=True)
class Patient:
    id: str
    name: str
    age: int
    conditions: List[str]


# Predefined dictionary of patients
PATIENTS: Dict[str, Patient] = {
    "P001": Patient(id="P001", name="Alice Johnson", age=34, conditions=["Hypertension"]),
    "P002": Patient(id="P002", name="Bob Smith", age=58, conditions=["Type 2 Diabetes"]),
    "P003": Patient(id="P003", name="Carla Gomez", age=42, conditions=["Asthma"]),
}


def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieve a Patient by patient ID.

    Args:
        patient_id: The unique identifier for the patient.

    Returns:
        The Patient object corresponding to the provided ID.

    Raises:
        ValueError: If the patient ID is not found in the predefined dictionary.
    """
    patient = PATIENTS.get(patient_id)
    if patient is None:
        raise ValueError(f"Patient ID not found: {patient_id}")
    return patient


# Allowed and restricted patient fields for template placeholders
ALLOWED_PATIENT_FIELDS: Set[str] = {"id", "name", "age"}
RESTRICTED_PATIENT_FIELDS: Set[str] = {"conditions"}


def validate_report_placeholders(template: str) -> bool:
    """
    Validate placeholders in a report template.

    Valid placeholders must reference patient attributes using the form:
      {patient.<field>}
    where <field> is one of the allowed patient fields.

    Examples of valid placeholders:
      {patient.name}, {patient.age}, {patient.id}

    Restricted placeholders (raise ValueError):
      {patient.conditions}

    Any other placeholder patterns or fields are considered invalid.

    Args:
        template: The template string containing placeholders.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or references a restricted field.
    """
    formatter = string.Formatter()
    invalid_placeholders: Set[str] = set()
    restricted_placeholders: Set[str] = set()

    for _, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is None:
            continue  # Literal text or escaped braces
        field = field_name.strip()

        # Allow format specs and conversions; validate only the field reference
        # e.g., for "{patient.age:03d}" -> field is "patient.age"
        # Split off any !conversion or :format_spec if they were embedded in field_name
        # (python's Formatter.parse already separates them, but be defensive)
        if "!" in field:
            field = field.split("!", 1)[0].strip()
        if ":" in field:
            field = field.split(":", 1)[0].strip()

        if field == "patient":
            invalid_placeholders.add("{patient}")
            continue

        if field.startswith("patient."):
            sub = field[len("patient.") :]

            # Disallow nested attribute/index access beyond a single attribute
            if not sub or any(ch in sub for ch in ".[]"):
                invalid_placeholders.add("{" + field_name + "}")
                continue

            attr = sub
            if attr in RESTRICTED_PATIENT_FIELDS:
                restricted_placeholders.add("{" + field_name + "}")
            elif attr in ALLOWED_PATIENT_FIELDS:
                # valid
                continue
            else:
                invalid_placeholders.add("{" + field_name + "}")
        else:
            # Any placeholder not referencing patient.* is invalid
            invalid_placeholders.add("{" + field_name + "}")

    if restricted_placeholders or invalid_placeholders:
        messages = []
        if restricted_placeholders:
            messages.append(
                "Restricted placeholders: " + ", ".join(sorted(restricted_placeholders))
            )
        if invalid_placeholders:
            messages.append(
                "Invalid placeholders: " + ", ".join(sorted(invalid_placeholders))
            )
        raise ValueError("; ".join(messages))

    return True
