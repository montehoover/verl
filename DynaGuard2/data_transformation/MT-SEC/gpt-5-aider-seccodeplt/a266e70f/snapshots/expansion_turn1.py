from dataclasses import dataclass
from typing import Dict


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


__all__ = ["Patient", "PATIENTS", "get_patient_by_id"]
