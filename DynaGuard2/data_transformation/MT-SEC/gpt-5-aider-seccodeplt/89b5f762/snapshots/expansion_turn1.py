from dataclasses import dataclass
from typing import Dict, List


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
