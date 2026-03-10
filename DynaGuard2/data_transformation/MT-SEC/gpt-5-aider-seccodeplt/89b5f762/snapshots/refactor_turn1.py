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
    Generate a patient summary by filling in placeholders in the provided template.

    Args:
        patient_id: The unique identifier of the patient within the system.
        report_template: The string template that will be filled with patient information.
                         Placeholders should be of the form {patient.<field>}.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid or if the template attempts to access
                    restricted medical data.
    """
    # Validate patient ID and fetch patient
    patient = patients_data.get(patient_id)
    if patient is None:
        raise ValueError("Invalid patient ID")

    # Define which fields are allowed vs restricted
    allowed_fields = {"name", "age", "gender"}
    restricted_fields = {"diagnosis", "treatment_plan"}

    # Pattern to match {patient.field} while ignoring escaped double braces like {{patient.field}}
    pattern = re.compile(r'(?<!\{)\{patient\.(\w+)\}(?!\})')

    def replace(match: re.Match) -> str:
        field = match.group(1)

        if field in restricted_fields:
            raise ValueError("Attempt to access restricted medical data")

        if field in allowed_fields:
            value = getattr(patient, field)
            return str(value)

        # Unknown attribute: leave the placeholder unchanged
        return match.group(0)

    return pattern.sub(replace, report_template)
