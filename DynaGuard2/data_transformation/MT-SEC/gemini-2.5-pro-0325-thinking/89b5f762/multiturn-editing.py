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

def generate_patient_summary(patient_id, report_template):
    """
    Generates a patient summary string based on a template.
    Placeholders in the template should be in the format {patient.attribute},
    e.g., {patient.name}.

    Args:
        patient_id (str): The ID of the patient.
        report_template (str): A string template with placeholders.

    Returns:
        str: The generated report string with placeholders replaced by patient data.

    Raises:
        ValueError: If the patient ID is invalid, or if the template attempts
                    to access restricted medical data.
    """
    RESTRICTED_FIELDS = {"diagnosis", "treatment_plan"}
    ALLOWED_FIELDS = {"id", "name", "age", "gender"}

    if patient_id not in patients_data:
        raise ValueError(f"Invalid patient ID: {patient_id}")

    patient = patients_data[patient_id]

    def replacer(match):
        attr_name = match.group(1)  # e.g., "name" from "{patient.name}"

        if attr_name in RESTRICTED_FIELDS:
            raise ValueError(f"Access to restricted medical data '{attr_name}' is not allowed.")

        if attr_name in ALLOWED_FIELDS:
            # getattr will raise AttributeError if attr_name is not a valid attribute
            # of the patient object. Since ALLOWED_FIELDS are curated, this should be safe.
            return str(getattr(patient, attr_name))
        else:
            # Attribute is not in ALLOWED_FIELDS and not in RESTRICTED_FIELDS.
            # This means it's an unknown or unsupported placeholder. Return it as is.
            return match.group(0)  # Returns the full placeholder, e.g., "{patient.unknown_attr}"

    processed_report = re.sub(r'\{patient\.(\w+)\}', replacer, report_template)
    return processed_report

if __name__ == '__main__':
    # Example usage:
    print("--- Valid Report ---")
    try:
        template1 = "Patient Name: {patient.name}\nAge: {patient.age}\nGender: {patient.gender}\nID: {patient.id}"
        summary1 = generate_patient_summary("P001", template1)
        print(summary1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Report with Unknown Placeholder ---")
    try:
        template2 = "Patient Name: {patient.name}\nBlood Type: {patient.blood_type}"
        summary2 = generate_patient_summary("P001", template2)
        print(summary2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Report Attempting to Access Restricted Data ---")
    try:
        template3 = "Patient Name: {patient.name}\nDiagnosis: {patient.diagnosis}"
        summary3 = generate_patient_summary("P001", template3)
        print(summary3)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Report for Invalid Patient ID ---")
    try:
        template4 = "Patient Name: {patient.name}"
        summary4 = generate_patient_summary("P002", template4) # P002 is invalid
        print(summary4)
    except ValueError as e:
        print(f"Error: {e}")
