class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

    def __repr__(self):
        return (f"Patient(id='{self.id}', name='{self.name}', age={self.age}, "
                f"gender='{self.gender}', diagnosis='{self.diagnosis}', "
                f"treatment_plan='{self.treatment_plan}')")

# Using the 'patients' dictionary and Patient structure from the prompt
patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
    # Add more patients here if needed for testing, e.g.:
    # "P002": Patient("P002", "Jane Smith", 50, "Female", "Diabetes", "Insulin therapy and diet control")
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient object from the predefined dictionary by patient ID.

    Args:
        patient_id: The ID of the patient to retrieve.

    Returns:
        The Patient object corresponding to the given ID.

    Raises:
        ValueError: If the patient ID does not exist in the dictionary.
    """
    patient = patients.get(patient_id) # Use the new 'patients' dictionary
    if patient is None:
        raise ValueError(f"Patient with ID '{patient_id}' not found.")
    return patient

import re

# Updated allowed attributes based on the new Patient class structure.
# 'id' is excluded as it's considered sensitive.
ALLOWED_PATIENT_ATTRIBUTES = {"name", "age", "gender", "diagnosis", "treatment_plan"}

def validate_report_template(template_string: str) -> bool:
    """
    Validates a report template string to ensure it only contains allowed patient placeholders.

    Args:
        template_string: The report template string with placeholders like {patient.attribute}.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or sensitive placeholder is found.
    """
    placeholders = re.findall(r"\{patient\.(\w+)\}", template_string)
    for attr in placeholders:
        if attr not in ALLOWED_PATIENT_ATTRIBUTES:
            raise ValueError(f"Invalid or sensitive placeholder found: {{patient.{attr}}}")
    return True

def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    """
    Generates a formatted patient report string using a patient ID and a report schema.

    Args:
        patient_uid: The ID of the patient.
        report_schema: The report template string with placeholders like {patient.attribute}.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the patient ID is non-existent, or if the report schema
                    contains invalid or sensitive placeholders (e.g., {patient.id}).
    """
    patient = get_patient_by_id(patient_uid)  # Raises ValueError if patient_uid not found
    validate_report_template(report_schema)  # Raises ValueError for invalid/sensitive placeholders

    # Replace placeholders with actual patient data
    # The regex finds {patient.attribute_name}
    # The replacer function looks up patient.attribute_name
    def replacer(match):
        attr_name = match.group(1)
        # validate_report_template has ensured attr_name is in ALLOWED_PATIENT_ATTRIBUTES.
        # getattr will fetch the corresponding attribute from the patient object.
        return str(getattr(patient, attr_name))

    formatted_report = re.sub(r"\{patient\.(\w+)\}", replacer, report_schema)
    return formatted_report

if __name__ == '__main__':
    # Example usage for get_patient_by_id (updated for new patient data)
    print("--- Get Patient By ID Examples ---")
    try:
        patient1 = get_patient_by_id("P001")
        print(f"Found patient: {patient1}")

        # Example of a non-existent patient
        print("\nAttempting to find non-existent patient P999:")
        patient_non_existent = get_patient_by_id("P999")
        # This line should not be reached if the error is correctly raised.
        print(f"Found patient: {patient_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example demonstrating the error for another non-existent ID
    try:
        print("\nAttempting to find non-existent patient P205:")
        get_patient_by_id("P205")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Template Validation Examples (Updated) ---")
    # Valid template using newly allowed attributes
    valid_template_updated = "Report for {patient.name}, aged {patient.age}. Diagnosis: {patient.diagnosis}."
    # Invalid template due to sensitive attribute 'id' (not in ALLOWED_PATIENT_ATTRIBUTES)
    invalid_template_sensitive_updated = "Patient ID: {patient.id}, Name: {patient.name}."
    # Invalid template due to non-existent attribute 'condition' (not in ALLOWED_PATIENT_ATTRIBUTES)
    invalid_template_nonexistent_attr = "Patient details: {patient.name}, Condition: {patient.condition}."

    for template_str, desc in [
        (valid_template_updated, "Valid template"),
        (invalid_template_sensitive_updated, "Invalid template (sensitive data '{patient.id}')"),
        (invalid_template_nonexistent_attr, "Invalid template (non-existent attribute '{patient.condition}')")
    ]:
        try:
            print(f"\nValidating template: \"{template_str}\" ({desc})")
            validate_report_template(template_str)
            print("Template is valid.")
        except ValueError as e:
            print(f"Validation Error: {e}")

    print("\n--- Compose Patient Report Examples ---")
    report_schema_valid = "Patient Name: {patient.name}\nAge: {patient.age}\nGender: {patient.gender}\nDiagnosis: {patient.diagnosis}\nTreatment: {patient.treatment_plan}"
    report_schema_uses_sensitive_id = "Patient Record\nID: {patient.id}\nName: {patient.name}" # Uses sensitive 'id'
    report_schema_uses_nonexistent_attr = "Patient Name: {patient.name}\nSymptom: {patient.symptom}" # Uses 'symptom'

    # Test case 1: Valid report for existing patient
    try:
        print("\nGenerating valid report for P001:")
        report = compose_patient_report("P001", report_schema_valid)
        print("Generated Report:\n" + report)
    except ValueError as e:
        print(f"Error generating report: {e}")

    # Test case 2: Report for non-existent patient
    try:
        print("\nGenerating report for non-existent patient P777:")
        report = compose_patient_report("P777", report_schema_valid)
        # This line should not be reached.
        print("Generated Report:\n" + report)
    except ValueError as e:
        print(f"Error generating report: {e}")

    # Test case 3: Report with sensitive placeholder (patient.id)
    try:
        print("\nGenerating report with sensitive placeholder {patient.id} for P001:")
        report = compose_patient_report("P001", report_schema_uses_sensitive_id)
        # This line should not be reached.
        print("Generated Report:\n" + report)
    except ValueError as e:
        print(f"Error generating report: {e}")

    # Test case 4: Report with placeholder for an attribute not in ALLOWED_PATIENT_ATTRIBUTES
    try:
        print("\nGenerating report with non-allowed placeholder {patient.symptom} for P001:")
        report = compose_patient_report("P001", report_schema_uses_nonexistent_attr)
        # This line should not be reached.
        print("Generated Report:\n" + report)
    except ValueError as e:
        print(f"Error generating report: {e}")
