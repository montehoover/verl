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

patients_data = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes"),
    "P002": Patient("P002", "Jane Smith", 32, "Female", "Asthma", "Inhaler and allergy management"),
    "P003": Patient("P003", "Robert Johnson", 58, "Male", "Diabetes", "Insulin therapy and diet control"),
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient object by their ID.

    Args:
        patient_id: The ID of the patient to retrieve.

    Returns:
        The Patient object corresponding to the given ID.

    Raises:
        ValueError: If the patient ID is not found in the database.
    """
    patient = patients_data.get(patient_id)
    if patient is None:
        raise ValueError(f"Patient with ID '{patient_id}' not found.")
    return patient

import re

# Allowed patient attributes for placeholders
ALLOWED_PATIENT_ATTRIBUTES = {"id", "name", "age", "gender", "diagnosis", "treatment_plan"}

def validate_report_placeholders(template: str) -> bool:
    """
    Validates placeholders in a report template.

    Placeholders should be in the format {patient.attribute}.
    Allowed attributes are 'name', 'age', 'patient_id'.

    Args:
        template: The report template string.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or restricted placeholder is found.
    """
    placeholders = re.findall(r"\{(patient\.[a-zA-Z_][a-zA-Z0-9_]*)\}", template)
    
    if not placeholders: # No placeholders found, template is valid by default
        return True

    for placeholder in placeholders:
        parts = placeholder.split('.')
        if len(parts) != 2 or parts[0] != "patient":
            raise ValueError(f"Invalid placeholder format: {{{placeholder}}}. Expected {{patient.attribute}}.")
        
        attribute_name = parts[1]
        if attribute_name not in ALLOWED_PATIENT_ATTRIBUTES:
            raise ValueError(f"Invalid or restricted placeholder attribute: {{{placeholder}}}. Allowed attributes are: {', '.join(ALLOWED_PATIENT_ATTRIBUTES)}.")
            
    return True

def generate_patient_summary(patient_id: str, report_template: str) -> str:
    """
    Generates a patient summary report by replacing placeholders with actual patient details.

    Args:
        patient_id: The ID of the patient.
        report_template: The report template string with placeholders.

    Returns:
        The formatted report string.

    Raises:
        ValueError: If the patient ID is invalid, or if the template contains
                    invalid or restricted placeholders.
    """
    # Step 1: Retrieve the patient. This will raise ValueError if patient_id is not found.
    patient = get_patient_by_id(patient_id)

    # Step 2: Validate placeholders. This will raise ValueError for invalid placeholders.
    validate_report_placeholders(report_template)

    # Step 3: Replace placeholders with patient data using re.sub
    def replacer(match):
        placeholder_key = match.group(1)  # This is "patient.attribute"
        attribute_name = placeholder_key.split('.')[1]
        # Validation has already confirmed attribute_name is allowed and exists.
        return str(getattr(patient, attribute_name))

    # This regex captures "patient.attribute" in group 1.
    # It's consistent with the one in validate_report_placeholders.
    placeholder_regex = r"\{(patient\.[a-zA-Z_][a-zA-Z0-9_]*)\}"
    
    formatted_report = re.sub(placeholder_regex, replacer, report_template)
    
    return formatted_report

if __name__ == '__main__':
    # Example Usage
    try:
        patient1 = get_patient_by_id("P001")
        print(f"Found patient: {patient1}")

        patient2 = get_patient_by_id("P002")
        print(f"Found patient: {patient2}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_id("P999")
        print(f"Found patient: {patient_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent patient to ensure the error is caught
        patient_non_existent_again = get_patient_by_id("P004")
        print(f"Found patient: {patient_non_existent_again}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example Usage for validate_report_placeholders
    print("\n--- Validating Report Placeholders ---")
    valid_template_1 = "Patient Name: {patient.name}, Age: {patient.age}"
    valid_template_2 = "Report for patient ID: {patient.id}." # Changed patient.patient_id to patient.id
    valid_template_3 = "No placeholders here."
    invalid_template_format = "Patient Info: {patient_name}" # Invalid format
    invalid_template_attr = "Patient Secret: {patient.secret_code}" # Invalid attribute
    invalid_template_object = "Details: {user.name}" # Invalid object
    
    templates_to_test = [
        ("Valid template 1", valid_template_1),
        ("Valid template 2", valid_template_2),
        ("Valid template 3 (no placeholders)", valid_template_3),
        ("Invalid template (format)", invalid_template_format),
        ("Invalid template (attribute)", invalid_template_attr),
        ("Invalid template (object)", invalid_template_object),
        ("Mixed valid/invalid", "Name: {patient.name}, Invalid: {patient.address}")
    ]

    for desc, template_str in templates_to_test:
        try:
            is_valid = validate_report_placeholders(template_str)
            print(f"'{desc}': Validation result: {is_valid} - Template: '{template_str}'")
        except ValueError as e:
            print(f"'{desc}': Validation Error: {e} - Template: '{template_str}'")

    # Example Usage for generate_patient_summary
    print("\n--- Generating Patient Summaries ---")
    
    summary_template_1 = "Patient: {patient.name} (ID: {patient.id})\nAge: {patient.age}, Gender: {patient.gender}\nDiagnosis: {patient.diagnosis}\nTreatment: {patient.treatment_plan}"
    summary_template_2 = "Patient Name: {patient.name}"
    summary_template_invalid_placeholder = "Patient Info: {patient.name}, Contact: {patient.phone_number}" # phone_number is invalid
    summary_template_valid_but_attr_missing_in_some_patient_if_any = "Patient: {patient.name}, Notes: {patient.notes}" # 'notes' not in ALLOWED_PATIENT_ATTRIBUTES

    patients_to_summarize = ["P001", "P002", "P999"] # P999 is non-existent
    templates_for_summary = [
        ("Full Summary", summary_template_1),
        ("Name Only", summary_template_2),
        ("Invalid Placeholder (phone_number)", summary_template_invalid_placeholder),
        ("Invalid Placeholder (notes)", summary_template_valid_but_attr_missing_in_some_patient_if_any)
    ]

    for patient_id_to_test in patients_to_summarize:
        print(f"\nTesting summaries for Patient ID: {patient_id_to_test}")
        for desc, template_str in templates_for_summary:
            try:
                report = generate_patient_summary(patient_id_to_test, template_str)
                print(f"--- {desc} for {patient_id_to_test} ---")
                print(report)
                print("--- End of Report ---")
            except ValueError as e:
                print(f"Error generating '{desc}' for {patient_id_to_test}: {e}")
