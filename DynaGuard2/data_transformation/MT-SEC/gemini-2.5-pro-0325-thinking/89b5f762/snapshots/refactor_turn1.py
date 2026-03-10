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
    Generates a patient summary based on a template.

    The template may contain placeholders such as '{patient.name}' or '{patient.age}',
    which will be replaced with corresponding patient details.

    Args:
        patient_id: The unique identifier of the patient within the system.
        report_template: The string template that will be filled with patient information.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid or if an attempt is made to access
                    restricted medical data (e.g., diagnosis, treatment_plan)
                    through the template.
    """
    if patient_id not in patients_data:
        raise ValueError(f"Invalid patient ID: {patient_id}")

    patient = patients_data[patient_id]

    # Find all attribute names requested in the template via {patient.attribute}
    # This allows checking for restricted attributes before attempting to format.
    # Example: re.findall(r"\{patient\.(\w+)\}", "{patient.name} {patient.age}") == ['name', 'age']
    requested_attributes = re.findall(r"\{patient\.(\w+)\}", report_template)

    restricted_attributes = {"diagnosis", "treatment_plan"}

    for attr_name in requested_attributes:
        if attr_name in restricted_attributes:
            raise ValueError(
                f"Attempt to access restricted medical data: template requests attribute '{attr_name}'"
            )

    # If no restricted attributes are requested, proceed to format the template.
    # The patient object itself is passed to .format(), allowing access to its attributes
    # like patient.name, patient.age, etc., directly in the template string.
    # If the template contains a placeholder for an attribute that does not exist
    # on the Patient object (e.g., {patient.non_existent_field}),
    # str.format() will raise an AttributeError. This is standard behavior
    # and not overridden by the specific ValueError conditions requested.
    summary = report_template.format(patient=patient)
    
    return summary
