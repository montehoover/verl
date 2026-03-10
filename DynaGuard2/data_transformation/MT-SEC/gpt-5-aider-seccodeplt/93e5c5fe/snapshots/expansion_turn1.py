from dataclasses import dataclass
from typing import Dict, Union

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
