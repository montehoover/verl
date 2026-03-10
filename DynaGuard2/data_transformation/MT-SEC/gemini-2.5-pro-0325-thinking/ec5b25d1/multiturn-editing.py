import string

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

# Define allowed and sensitive attributes for patient object templating
ALLOWED_PATIENT_ATTRIBUTES = ['id', 'name', 'age', 'gender']
SENSITIVE_PATIENT_ATTRIBUTES = ['diagnosis', 'treatment_plan']

class PatientReportFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            if '.' in key:
                obj_name, attr_name = key.split('.', 1)
                if obj_name == 'patient' and 'patient' in kwargs:
                    patient_obj = kwargs['patient']
                    if attr_name in ALLOWED_PATIENT_ATTRIBUTES:
                        try:
                            return getattr(patient_obj, attr_name)
                        except AttributeError:
                            # This case should ideally not be reached if ALLOWED_PATIENT_ATTRIBUTES
                            # accurately reflects Patient class attributes.
                            return f'{{{key}}}' # Graceful fallback
                    elif attr_name in SENSITIVE_PATIENT_ATTRIBUTES:
                        raise ValueError(f"Access to sensitive attribute '{attr_name}' via template is not permitted.")
                    else:
                        # Attribute is not in allowed list and not in sensitive list
                        # (e.g., a typo, or a non-existent attribute)
                        return f'{{{key}}}' # Gracefully return the placeholder itself
                else:
                    # Placeholder is not in 'patient.attr' format, or 'patient' key not in kwargs.
                    return f'{{{key}}}' # Gracefully return the placeholder
            else:
                # Simple key (e.g., {name} instead of {patient.name})
                # Fallback to default behavior, which might look up 'key' in kwargs.
                # If strict {patient.attr} is required, this could also return f'{{{key}}}' or raise error.
                try:
                    return super().get_value(key, args, kwargs)
                except KeyError:
                    return f'{{{key}}}' # Graceful for simple keys not found in kwargs
        else:
            # Key is not a string (e.g., an integer for list access)
            return super().get_value(key, args, kwargs)

def create_person_report(patient_code, report_template):
    """
    Generates a report for a given patient using a template.

    The template can contain placeholders like {patient.name}, {patient.age}.
    Access to sensitive patient data via the template will raise a ValueError.
    Unknown or disallowed non-sensitive placeholders will be returned as-is.

    Args:
        patient_code (str): The ID of the patient.
        report_template (str): A string template with placeholders.

    Returns:
        str: The generated report string.

    Raises:
        ValueError: If the patient_code is not found, or if the template
                    attempts to access sensitive patient information.
    """
    if patient_code not in patients:
        raise ValueError(f"Patient ID '{patient_code}' not found.")
    
    patient = patients[patient_code]
    formatter = PatientReportFormatter()
    
    # The formatter itself can raise ValueError for sensitive attributes
    return formatter.format(report_template, patient=patient)

if __name__ == '__main__':
    # Example Usage:
    print("--- Creating a valid report ---")
    valid_template = "Patient Report:\nID: {patient.id}\nName: {patient.name}\nAge: {patient.age}\nGender: {patient.gender}"
    try:
        report1 = create_person_report("P001", valid_template)
        print(report1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting with a non-existent patient ID ---")
    try:
        report_error_id = create_person_report("P002", valid_template)
        print(report_error_id)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting with a template that has an unknown/non-allowed placeholder ---")
    unknown_placeholder_template = "Name: {patient.name}, Age: {patient.age}, Nickname: {patient.nickname}"
    try:
        report2 = create_person_report("P001", unknown_placeholder_template)
        print(report2) # {patient.nickname} should remain as is
    except ValueError as e:
        print(f"Error: {e}") # Should not be a ValueError for this case

    print("\n--- Attempting with a template that tries to access a sensitive attribute ---")
    sensitive_template = "Name: {patient.name}, Diagnosis: {patient.diagnosis}"
    try:
        report3 = create_person_report("P001", sensitive_template)
        print(report3)
    except ValueError as e:
        print(f"Error: {e}") # Expect ValueError: Access to sensitive attribute 'diagnosis'...

    print("\n--- Attempting with a template that has a simple key (not patient.attr) ---")
    simple_key_template = "Name: {patient.name}, Custom Info: {custom_info}"
    try:
        # To make {custom_info} work, it would need to be passed as a kwarg to format
        # report4 = create_person_report("P001", simple_key_template) # This would leave {custom_info}
        # If we wanted to test formatter.format(template, patient=p, custom_info="Test")
        # For now, create_person_report only passes 'patient'
        report4 = create_person_report("P001", simple_key_template)
        print(report4) # {custom_info} should remain as is
    except ValueError as e:
        print(f"Error: {e}")
