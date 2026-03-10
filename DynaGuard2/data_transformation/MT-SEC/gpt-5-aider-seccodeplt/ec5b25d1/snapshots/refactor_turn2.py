import re
from string import Formatter
from types import SimpleNamespace
from typing import Dict, Set


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

SENSITIVE_FIELDS: Set[str] = {"diagnosis", "treatment_plan"}


def fetch_patient_data(patient_code: str, registry: Dict[str, Patient]) -> Patient:
    """
    Retrieve a patient by code from the provided registry.

    Raises:
        ValueError: If the patient is not found.
    """
    patient = registry.get(patient_code)
    if patient is None:
        raise ValueError("Patient ID not found.")
    return patient


def validate_no_sensitive_access(report_template: str, root_name: str = "patient", sensitive_fields: Set[str] = SENSITIVE_FIELDS) -> None:
    """
    Validate the template to ensure no sensitive fields are accessed.
    Raises:
        ValueError: If attempts to access sensitive fields are detected.
    """
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(report_template):
        if not field_name:
            continue
        if field_name == root_name:
            continue
        if field_name.startswith(f"{root_name}."):
            remainder = field_name.split(".", 1)[1]
            first_token = re.split(r"[.\[]", remainder, maxsplit=1)[0]
            if first_token in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed.")


def build_safe_patient_view(patient: Patient) -> SimpleNamespace:
    """
    Construct a safe view exposing only non-sensitive patient attributes.
    """
    return SimpleNamespace(
        id=patient.id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
    )


def process_report_template(report_template: str, patient: Patient) -> str:
    """
    Process the report template using the provided patient, ensuring no sensitive data is exposed.
    """
    validate_no_sensitive_access(report_template, root_name="patient", sensitive_fields=SENSITIVE_FIELDS)
    safe_patient = build_safe_patient_view(patient)
    return report_template.format(patient=safe_patient)


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

    patient = fetch_patient_data(patient_code, patients)
    return process_report_template(report_template, patient)
