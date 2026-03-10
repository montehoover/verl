import re

class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

patients_data = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

def generate_patient_summary(patient_id: str, report_template: str) -> str:
    """
    Generate a patient summary by filling a template with patient information.

    Args:
        patient_id: The unique identifier of the patient within the system.
        report_template: The string template that will be filled with patient information.
                         Placeholders should be in the form {patient.<field>}.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid, the template references an unknown field,
                    or if an attempt is made to access restricted medical data (diagnosis or treatment_plan).
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Invalid patient ID.")
    if patient_id not in patients_data:
        raise ValueError("Invalid patient ID.")

    patient = patients_data[patient_id]

    # Find all patient placeholders like {patient.name}
    placeholder_pattern = re.compile(r"\{patient\.([a-zA-Z_][a-zA-Z0-9_]*)\}")
    fields_requested = set(placeholder_pattern.findall(report_template))

    # Restricted medical data fields that must not be exposed
    restricted_fields = {"diagnosis", "treatment_plan"}

    # Validate requested fields
    for field in fields_requested:
        if field in restricted_fields:
            raise ValueError("Access to restricted medical data is not allowed.")
        if not hasattr(patient, field):
            raise ValueError(f"Invalid placeholder field referenced: '{field}'")

    # Perform replacement
    def _replace(match: re.Match) -> str:
        field = match.group(1)
        value = getattr(patient, field, "")
        return str(value)

    return placeholder_pattern.sub(_replace, report_template)
