import re
from string import Formatter
from types import SimpleNamespace
from typing import Dict


class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


patients: Dict[str, Patient] = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}


def create_person_report(patient_code: str, report_template: str) -> str:
    """
    Apply a report template to a patient record, substituting non-sensitive fields.

    Args:
        patient_code: The identification code for the patient.
        report_template: A string template with placeholders like '{patient.name}'.

    Returns:
        A string with the patient's non-sensitive information formatted as per the template.

    Raises:
        ValueError: If patient is not found, or if the template attempts to access
                    sensitive information (diagnosis or treatment_plan).
    """
    if not isinstance(patient_code, str):
        raise ValueError("patient_code must be a string.")
    if not isinstance(report_template, str):
        raise ValueError("report_template must be a string.")

    patient = patients.get(patient_code)
    if patient is None:
        raise ValueError("Patient ID not found.")

    sensitive_fields = {"diagnosis", "treatment_plan"}

    # Pre-scan the template to detect attempts to access sensitive fields.
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(report_template):
        if not field_name:
            continue
        # Only care about fields that reference the 'patient' object.
        if field_name == "patient":
            # Not sensitive by itself; allowed (will produce object repr if used).
            continue
        if field_name.startswith("patient."):
            # Extract the first attribute token after 'patient.'
            remainder = field_name.split(".", 1)[1]
            # Tokenize by next '.' or '[' to get the immediate attribute name.
            first_token = re.split(r"[.\[]", remainder, maxsplit=1)[0]
            if first_token in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed.")

    # Provide a safe view of the patient that only exposes non-sensitive attributes.
    safe_patient = SimpleNamespace(
        id=patient.id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
    )

    # Perform formatting. If the template contains unsupported fields, this will raise,
    # which is acceptable per the specification (only sensitive access must be ValueError).
    return report_template.format(patient=safe_patient)
