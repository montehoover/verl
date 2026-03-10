from typing import Dict
from string import Formatter


class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan


# Predefined dictionary of patients keyed by patient code
patients: Dict[str, Patient] = {
    "P001": Patient(
        "P001",
        "John Doe",
        45,
        "Male",
        "Hypertension",
        "Medication and lifestyle changes",
    )
}


def get_patient_by_code(patient_code: str) -> Patient:
    """
    Return the Patient corresponding to the given patient_code.

    Raises:
        ValueError: If the patient_code does not exist in the predefined dictionary.
    """
    try:
        return patients[patient_code]
    except KeyError:
        raise ValueError(f"Patient code not found: {patient_code}")


# Allowed non-sensitive placeholders in report templates
ALLOWED_TEMPLATE_PLACEHOLDERS = {
    "patient.name",
    "patient.age",
}


def validate_report_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed non-sensitive placeholders.

    Allowed placeholders:
      - {patient.name}
      - {patient.age}

    Returns:
        bool: True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or sensitive placeholder is detected.
    """
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if not field_name:
            # No placeholder in this segment (or escaped braces), skip
            continue
        if field_name not in ALLOWED_TEMPLATE_PLACEHOLDERS:
            raise ValueError(f"Invalid or sensitive placeholder: {{{field_name}}}")
    return True


def create_person_report(patient_code: str, report_template: str) -> str:
    """
    Generate a formatted patient report from a template.

    Args:
        patient_code (str): The patient code used to look up the patient.
        report_template (str): The template string containing placeholders
                               like {patient.name} and {patient.age}.

    Returns:
        str: The formatted report string.

    Raises:
        ValueError: If the patient code does not exist or the template contains
                    invalid/sensitive placeholders.
    """
    # Validate template for sensitive placeholders
    validate_report_template(report_template)

    # Retrieve patient (raises ValueError if not found)
    patient = get_patient_by_code(patient_code)

    # Perform formatting using safe, allowed fields
    try:
        return report_template.format(patient=patient)
    except Exception as exc:
        # Wrap any formatting error into a ValueError for a consistent API
        raise ValueError(f"Unable to format report: {exc}")
