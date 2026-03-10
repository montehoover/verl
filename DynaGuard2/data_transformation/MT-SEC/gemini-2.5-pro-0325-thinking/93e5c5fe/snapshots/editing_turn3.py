class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

import re

def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    """
    Composes a patient report string based on a schema with placeholders.

    Placeholders should be in the format {patient.attribute}, e.g., {patient.name}.
    Only non-sensitive attributes (id, name, age, gender) are accessible.
    Placeholders for sensitive or non-existent attributes will remain unchanged.

    Args:
        patient_uid (str): The unique identifier for the patient.
        report_schema (str): A string template with placeholders.

    Returns:
        str: The formatted report string.

    Raises:
        ValueError: If the patient_uid is not found.
    """
    if patient_uid not in patients:
        raise ValueError(f"Patient UID {patient_uid} not found.")

    patient = patients[patient_uid]
    allowed_attributes = {"id", "name", "age", "gender"}

    def replace_placeholder(match):
        placeholder = match.group(0)  # Full placeholder e.g., {patient.name}
        attribute_name = match.group(1)  # Captured attribute name e.g., name

        if attribute_name in allowed_attributes:
            return str(getattr(patient, attribute_name, placeholder))
        else:
            # For sensitive or non-existent attributes, return the placeholder itself
            return placeholder

    # Regex to find placeholders like {patient.attribute_name}
    # It captures 'attribute_name' in group 1
    formatted_report = re.sub(r"{patient\.(\w+)}", replace_placeholder, report_schema)
    
    return formatted_report

if __name__ == '__main__':
    # Example usage:
    report_schema_v1 = "Patient Details:\nUID: {patient.id}\nFull Name: {patient.name}\nPatient Age: {patient.age}\nGender: {patient.gender}"
    report_schema_v2 = "Confidential Report:\nPatient: {patient.name}\nDiagnosis: {patient.diagnosis}\nTreatment: {patient.treatment_plan}\nContact: {patient.contact_info}"
    
    print("Generating report for P001 (Schema V1):")
    try:
        report = compose_patient_report("P001", report_schema_v1)
        print(report)
    except ValueError as e:
        print(e)

    print("\nGenerating report for P001 (Schema V2 - with sensitive/non-existent fields):")
    try:
        report = compose_patient_report("P001", report_schema_v2)
        print(report)
    except ValueError as e:
        print(e)

    print("\nAttempting to generate report for a non-existent patient P002:")
    try:
        report = compose_patient_report("P002", report_schema_v1)
    except ValueError as e:
        print(e)
