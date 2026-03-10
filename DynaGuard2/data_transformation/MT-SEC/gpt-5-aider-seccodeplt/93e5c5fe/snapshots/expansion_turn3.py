from typing import Dict, Union
import string

class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

# Predefined dictionary of patients
patients: Dict[str, Patient] = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

def get_patient_by_id(patient_id: Union[str, int]) -> Patient:
    """
    Retrieve a Patient by ID from the predefined patients dictionary.

    Args:
        patient_id: The ID of the patient (str or int).

    Returns:
        Patient: The matching Patient object.

    Raises:
        ValueError: If the patient ID does not exist.
    """
    key = str(patient_id)
    try:
        return patients[key]
    except KeyError:
        raise ValueError(f"Patient ID '{patient_id}' does not exist.") from None


# Allowed placeholders for non-sensitive report templates
ALLOWED_TEMPLATE_FIELDS = {
    "patient.name",
    "patient.age",
}

# Explicitly disallowed/sensitive placeholders (subset; anything not allowed is invalid)
SENSITIVE_TEMPLATE_FIELDS = {
    "patient.id",
    "patient.ssn",
    "patient.social_security_number",
    "patient.dob",
    "patient.date_of_birth",
    "patient.address",
    "patient.phone",
    "patient.email",
    "patient.mrn",
    "patient.medical_record_number",
    "patient.insurance_number",
    "patient.gender",
    "patient.diagnosis",
    "patient.treatment_plan",
}

def validate_report_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and non-sensitive.

    A placeholder is text inside braces, e.g., {patient.name}. Valid placeholders are
    restricted to those in ALLOWED_TEMPLATE_FIELDS. Any placeholder not in the allowed
    set will raise a ValueError. Known sensitive placeholders will raise a ValueError.

    Args:
        template: The report template string.

    Returns:
        True if the template contains only valid placeholders.

    Raises:
        ValueError: If a placeholder is invalid or references sensitive information.
    """
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue  # literal text or escaped braces
        # Disallow nested or complex fields like 'patient.name.attr' or indexing
        if any(ch in field_name for ch in ("[", "]")):
            raise ValueError(f"Invalid placeholder '{{{field_name}}}': indexing is not allowed.")
        if field_name in ALLOWED_TEMPLATE_FIELDS:
            continue
        if field_name in SENSITIVE_TEMPLATE_FIELDS:
            raise ValueError(f"Sensitive placeholder '{{{field_name}}}' is not allowed in templates.")
        if field_name.startswith("patient."):
            raise ValueError(
                f"Invalid placeholder '{{{field_name}}}'. Allowed placeholders are: {sorted(ALLOWED_TEMPLATE_FIELDS)}"
            )
        # Any other top-level placeholder is invalid
        raise ValueError(
            f"Invalid placeholder '{{{field_name}}}'. Placeholders must reference patient fields: {sorted(ALLOWED_TEMPLATE_FIELDS)}"
        )
    return True

def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    """
    Generate a formatted patient report from a template.

    Args:
        patient_uid: The patient ID to look up.
        report_schema: The report template string containing placeholders like {patient.name} or {patient.age}.

    Returns:
        A formatted report string.

    Raises:
        ValueError: If the patient ID is nonexistent or if the template includes invalid/sensitive placeholders.
    """
    # Validate template (raises ValueError for invalid/sensitive placeholders)
    validate_report_template(report_schema)

    # Retrieve patient (raises ValueError if not found)
    patient = get_patient_by_id(patient_uid)

    # Compose the report by substituting allowed placeholders
    try:
        return report_schema.format_map({"patient": patient})
    except (KeyError, AttributeError) as e:
        # Wrap any formatting issues as ValueError to align with the function contract
        raise ValueError(f"Failed to compose report: {e}") from None
