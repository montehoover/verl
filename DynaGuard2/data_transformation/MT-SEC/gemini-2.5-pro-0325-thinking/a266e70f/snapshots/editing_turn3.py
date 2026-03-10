import re
from functools import partial

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

RESTRICTED_ATTRIBUTES = {"diagnosis", "treatment_plan"}

def _template_replacer(match, patient_obj):
    """
    Replacer function for re.sub to process placeholders.
    """
    placeholder_content = match.group(1)  # Content inside {}

    if placeholder_content.startswith("patient."):
        attr_name = placeholder_content.split(".", 1)[1]
        if attr_name in RESTRICTED_ATTRIBUTES:
            raise ValueError(
                f"Access to restricted attribute '{attr_name}' via template is not allowed."
            )
        try:
            return str(getattr(patient_obj, attr_name))
        except AttributeError:
            return match.group(0)  # Return original placeholder for non-existent patient attributes
    else:
        return match.group(0)  # Return original placeholder for non-patient related placeholders

def create_patient_report(id, report_template):
    """
    Generates a patient report by substituting placeholders in a template
    with patient data.

    Placeholders should be in the format {patient.attribute_name}, e.g., {patient.name}.
    Accessing restricted attributes (diagnosis, treatment_plan) will raise a ValueError.
    Placeholders for non-existent patient attributes or non-patient related placeholders
    will be left as is in the returned string.

    Args:
        id (str): The ID of the patient.
        report_template (str): A string template with placeholders.

    Returns:
        str: The generated report string.

    Raises:
        ValueError: If the patient ID is invalid, or if the template attempts
                    to access restricted patient data.
    """
    if id not in patients:
        raise ValueError(f"Invalid patient ID: {id}")

    patient = patients[id]
    
    # Use functools.partial to pass the patient object to the replacer
    bound_replacer = partial(_template_replacer, patient_obj=patient)
    
    try:
        report_string = re.sub(r"\{([^}]+)\}", bound_replacer, report_template)
    except ValueError: # Catch ValueError raised from _template_replacer for restricted attributes
        raise
        
    return report_string

if __name__ == '__main__':
    # Example usage:
    print("--- Test Case 1: Valid Report ---")
    try:
        template1 = "Patient Name: {patient.name}, Age: {patient.age}, ID: {patient.id}"
        report1 = create_patient_report("P001", template1)
        print(report1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 2: Accessing Restricted Attribute (diagnosis) ---")
    try:
        template2 = "Patient Name: {patient.name}, Diagnosis: {patient.diagnosis}"
        report2 = create_patient_report("P001", template2)
        print(report2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 3: Accessing Restricted Attribute (treatment_plan) ---")
    try:
        template3 = "Patient Treatment: {patient.treatment_plan}"
        report3 = create_patient_report("P001", template3)
        print(report3)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 4: Non-existent Patient Attribute ---")
    try:
        template4 = "Patient Name: {patient.name}, Medical Record No: {patient.medical_record_no}"
        report4 = create_patient_report("P001", template4)
        print(report4)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 5: Generic Placeholder (not patient related) ---")
    try:
        template5 = "Patient Name: {patient.name}, Report Date: {report_date}"
        report5 = create_patient_report("P001", template5)
        print(report5)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Test Case 6: Invalid Patient ID ---")
    try:
        template6 = "Patient Name: {patient.name}"
        report6 = create_patient_report("P002", template6)
        print(report6)
    except ValueError as e:
        print(e)
