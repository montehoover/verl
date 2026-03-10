import re
from dataclasses import dataclass

@dataclass
class Patient:
    """Represents a patient."""
    patient_id: str
    name: str
    age: int
    gender: str
    diagnosis: str
    treatment_plan: str

# Predefined dictionary of patients
# In a real application, this data would likely come from a database or external service.
_patients_database = {
    "P001": Patient(patient_id="P001", name="John Doe", age=45, gender="Male", diagnosis="Hypertension", treatment_plan="Medication and lifestyle changes"),
    "P002": Patient(patient_id="P002", name="Jane Smith", age=32, gender="Female", diagnosis="Diabetes", treatment_plan="Insulin therapy and diet control"),
    "P003": Patient(patient_id="P003", name="Alice Brown", age=58, gender="Female", diagnosis="Arthritis", treatment_plan="Pain management and physical therapy"),
    "P004": Patient(patient_id="P004", name="Bob Johnson", age=67, gender="Male", diagnosis="Asthma", treatment_plan="Inhaler and avoidance of triggers"),
}

def get_patient_by_id(patient_id: str) -> Patient:
    """
    Retrieves a patient's details by their ID.

    Args:
        patient_id: The ID of the patient to retrieve.

    Returns:
        The Patient object corresponding to the given ID.

    Raises:
        ValueError: If the patient ID is not found in the database.
    """
    patient = _patients_database.get(patient_id)
    if patient is None:
        raise ValueError(f"Patient with ID '{patient_id}' not found.")
    return patient

def validate_report_placeholders(template_string: str) -> bool:
    """
    Validates placeholders in a report template string.

    Placeholders should be in the format {patient.attribute}, where 'attribute'
    is a valid attribute of the Patient class.

    Args:
        template_string: The report template string containing placeholders.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid (e.g., refers to a non-existent
                    attribute or is improperly formatted) or refers to a restricted attribute.
    """
    # Valid attributes are the fields of the Patient dataclass
    valid_attributes = set(Patient.__annotations__.keys())

    # Find all placeholders like {patient.attribute_name}
    placeholders = re.findall(r"\{patient\.(\w+)\}", template_string)

    if not placeholders and "{" in template_string: # Handles cases like "{patient}" or "{patient..name}"
        # Check for malformed placeholders if any curly brace is present but no valid pattern was found
        if re.search(r"\{patient[^\w\s\.\}]*\}", template_string) or \
           re.search(r"\{patient\.\.[^\}]*\}", template_string) or \
           re.search(r"\{patient\.\}", template_string):
            raise ValueError("Invalid placeholder format found in template.")

    for attr in placeholders:
        if attr not in valid_attributes:
            raise ValueError(f"Invalid placeholder attribute: '{attr}'. "
                             f"Valid attributes are: {', '.join(sorted(list(valid_attributes)))}.")
        # Add any restricted attribute checks here if needed in the future
        # For example:
        # if attr == "some_restricted_field":
        #     raise ValueError(f"Placeholder attribute '{attr}' is restricted.")

    return True

def create_patient_report(patient_id: str, report_template: str) -> str:
    """
    Generates a patient report by replacing placeholders in a template with patient details.

    Args:
        patient_id: The ID of the patient for whom the report is generated.
        report_template: The template string for the report, containing placeholders
                         like {patient.name}, {patient.age}, etc.

    Returns:
        The formatted report string with placeholders replaced by actual patient data.

    Raises:
        ValueError: If the patient ID is invalid, or if the template contains
                    invalid or restricted placeholders.
    """
    patient = get_patient_by_id(patient_id)
    validate_report_placeholders(report_template) # This will raise ValueError if invalid

    def replace_placeholder(match):
        attr_name = match.group(1)
        # getattr is safe here because validate_report_placeholders already confirmed attr_name is valid
        return str(getattr(patient, attr_name))

    report = re.sub(r"\{patient\.(\w+)\}", replace_placeholder, report_template)
    return report

if __name__ == '__main__':
    # Example usage:
    try:
        patient1 = get_patient_by_id("P001")
        print(f"Found patient: {patient1.name}, Age: {patient1.age}, Condition: {patient1.condition}")

        patient2 = get_patient_by_id("P003")
        print(f"Found patient: {patient2.name}, Age: {patient2.age}, Condition: {patient2.condition}")

        # Example of a non-existent patient
        patient_non_existent = get_patient_by_id("P999")
        print(patient_non_existent) # This line won't be reached
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent patient
        patient_non_existent_2 = get_patient_by_id("P005")
        print(patient_non_existent_2) # This line won't be reached
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Validating Report Placeholders ---")
    # Example usage for validate_report_placeholders:
    valid_template_1 = "Patient Name: {patient.name}, Age: {patient.age}"
    valid_template_2 = "Condition: {patient.condition}"
    valid_template_3 = "Patient ID: {patient.patient_id}. This report is for {patient.name}."
    valid_template_no_placeholders = "This is a static report."

    invalid_template_1 = "Patient Name: {patient.name}, Secret: {patient.secret_code}"
    invalid_template_2 = "Details: {patient.non_existent_field}"
    invalid_template_format_1 = "Patient: {patient_name}" # incorrect format
    invalid_template_format_2 = "Patient: {patient..name}" # incorrect format
    invalid_template_format_3 = "Patient: {patient.}" # incorrect format


    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2": valid_template_2,
        "Valid Template 3": valid_template_3,
        "Valid Template (No Placeholders)": valid_template_no_placeholders,
        "Invalid Template (Unknown Attribute)": invalid_template_1,
        "Invalid Template (Non-existent Field)": invalid_template_2,
        "Invalid Template (Format 1)": invalid_template_format_1,
        "Invalid Template (Format 2)": invalid_template_format_2,
        "Invalid Template (Format 3)": invalid_template_format_3,
    }

    for name, template in templates_to_test.items():
        try:
            is_valid = validate_report_placeholders(template)
            print(f"'{name}': Validation result: {is_valid}")
        except ValueError as e:
            print(f"'{name}': Validation Error: {e}")

    print("\n--- Generating Patient Reports ---")
    # Example usage for create_patient_report:
    report_template_1 = "Patient Report:\nID: {patient.patient_id}\nName: {patient.name}\nAge: {patient.age}\nGender: {patient.gender}\nDiagnosis: {patient.diagnosis}\nTreatment: {patient.treatment_plan}"
    report_template_2 = "Summary for {patient.name} ({patient.patient_id}): Diagnosed with {patient.diagnosis}."
    invalid_attr_template = "Patient Name: {patient.name}, MedicalRecordID: {patient.medical_id}" # medical_id is not a valid field

    # Test successful report generation
    try:
        report1 = create_patient_report("P001", report_template_1)
        print("\nGenerated Report 1:\n", report1)
    except ValueError as e:
        print(f"\nError generating report 1: {e}")

    try:
        report2 = create_patient_report("P002", report_template_2)
        print("\nGenerated Report 2:\n", report2)
    except ValueError as e:
        print(f"\nError generating report 2: {e}")

    # Test with invalid patient ID
    try:
        print("\nAttempting report for non-existent patient P999:")
        create_patient_report("P999", report_template_1)
    except ValueError as e:
        print(f"Error: {e}")

    # Test with invalid placeholder in template
    try:
        print("\nAttempting report with invalid placeholder template:")
        create_patient_report("P003", invalid_attr_template)
    except ValueError as e:
        print(f"Error: {e}")

    # Test with malformed placeholder
    malformed_template = "Patient: {patient..name}"
    try:
        print("\nAttempting report with malformed placeholder template:")
        create_patient_report("P004", malformed_template)
    except ValueError as e:
        print(f"Error: {e}")
