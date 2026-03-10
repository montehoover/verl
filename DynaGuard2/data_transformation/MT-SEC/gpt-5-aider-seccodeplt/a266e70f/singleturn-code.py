import re
from typing import Dict


class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


# Example in-memory patient store
patients: Dict[str, Patient] = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


def create_patient_report(id: str, report_template: str) -> str:
    """
    Create a patient report by filling a template with basic patient information.

    Args:
        id: The unique identifier of the patient within the system.
        report_template: The string template containing placeholders like '{patient.name}'.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid or if the template attempts
                    to access restricted medical data.
    """
    if not isinstance(id, str) or not id.strip():
        raise ValueError("Invalid patient ID.")

    if not isinstance(report_template, str):
        raise ValueError("report_template must be a string.")

    try:
        patient = patients[id]
    except KeyError:
        raise ValueError("Invalid patient ID.")

    # Define which fields are allowed in reports (basic info only).
    allowed_fields = {"name", "age", "gender"}

    # Find all placeholders of the form {patient.<field>}
    pattern = re.compile(r"\{patient\.([a-zA-Z_][a-zA-Z0-9_]*)\}")

    # Validate that no restricted fields are being accessed
    placeholders = pattern.findall(report_template)
    for field in placeholders:
        if field not in allowed_fields:
            # Access to restricted or unknown fields is not permitted
            raise ValueError("Attempt to access restricted medical data.")

    # Replace allowed placeholders with actual values
    def replace(match: re.Match) -> str:
        field = match.group(1)
        value = getattr(patient, field, None)
        # Field existence already validated by allowed_fields; this is a safeguard.
        if value is None:
            raise ValueError("Attempt to access restricted medical data.")
        return str(value)

    return pattern.sub(replace, report_template)
