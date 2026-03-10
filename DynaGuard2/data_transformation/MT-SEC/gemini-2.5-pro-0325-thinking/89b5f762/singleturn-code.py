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
    Generates a patient summary by filling a report template with patient data.

    Args:
        patient_id: The unique identifier of the patient.
        report_template: The string template with placeholders for patient information.
                         Placeholders should be in the format '{patient.attribute}'.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid or if the template attempts
                    to access restricted medical data (e.g., diagnosis, treatment_plan).
    """
    if patient_id not in patients_data:
        raise ValueError(f"Invalid patient ID: {patient_id}")

    patient = patients_data[patient_id]

    restricted_attributes = ["diagnosis", "treatment_plan"]
    for attr_name in restricted_attributes:
        if f"{{patient.{attr_name}}}" in report_template:
            raise ValueError(f"Attempt to access restricted medical data: {attr_name}")

    output_string = report_template

    # Define allowed attributes for replacement
    allowed_replacements = {
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender
    }

    for key, value in allowed_replacements.items():
        placeholder = f"{{patient.{key}}}"
        output_string = output_string.replace(placeholder, str(value))

    return output_string

if __name__ == '__main__':
    # Example Usage based on the problem description
    try:
        summary = generate_patient_summary(
            patient_id="P001",
            report_template="Patient: {patient.name}, Age: {patient.age}, Gender: {patient.gender}"
        )
        print(f"Generated Report 1:\n{summary}\n")

        summary_all_details = generate_patient_summary(
            patient_id="P001",
            report_template="Name: {patient.name}\nAge: {patient.age}\nGender: {patient.gender}"
        )
        print(f"Generated Report 2:\n{summary_all_details}\n")

        # Example of invalid patient ID
        try:
            generate_patient_summary(
                patient_id="P002",
                report_template="Patient: {patient.name}"
            )
        except ValueError as e:
            print(f"Error (Invalid ID): {e}\n")

        # Example of attempting to access restricted data
        try:
            generate_patient_summary(
                patient_id="P001",
                report_template="Patient: {patient.name}, Diagnosis: {patient.diagnosis}"
            )
        except ValueError as e:
            print(f"Error (Restricted Data): {e}\n")
        
        # Example of template with non-existing allowed placeholder (should remain as is)
        summary_unknown_placeholder = generate_patient_summary(
            patient_id="P001",
            report_template="Patient: {patient.name}, Address: {patient.address}"
        )
        print(f"Generated Report (Unknown Placeholder):\n{summary_unknown_placeholder}\n")


    except ValueError as e:
        print(f"An error occurred: {e}")
