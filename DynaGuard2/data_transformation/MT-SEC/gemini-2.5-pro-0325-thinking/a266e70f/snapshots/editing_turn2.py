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

def get_formatted_patient_details(patient_id, format_template):
    """
    Formats patient details (ID, name, age, gender) for a given patient ID using a template.

    Args:
        patient_id (str): The ID of the patient.
        format_template (str): A string template with placeholders like {patient_id}, 
                               {patient_name}, {patient_age}, {patient_gender}.

    Returns:
        str: The formatted string with patient details.

    Raises:
        ValueError: If the patient ID is invalid.
    """
    if patient_id not in patients:
        raise ValueError(f"Invalid patient ID: {patient_id}")
    
    patient = patients[patient_id]
    
    details = {
        "patient_id": patient.id,
        "patient_name": patient.name,
        "patient_age": patient.age,
        "patient_gender": patient.gender
    }
    
    # Accessing diagnosis or treatment_plan here could be restricted.
    # This function only exposes basic details through the 'details' dictionary.
    # If an attempt was made to access patient.diagnosis or patient.treatment_plan
    # without proper authorization, a ValueError should be raised (though not directly
    # handled by this function's template mechanism for fields not in 'details').

    template = string.Template(format_template)
    return template.safe_substitute(details)

if __name__ == '__main__':
    # Example usage:
    try:
        template1 = "Patient Name: {patient_name}, Age: {patient_age}, ID: {patient_id}"
        formatted_details1 = get_formatted_patient_details("P001", template1)
        print("Formatted Details 1:")
        print(formatted_details1)

        template2 = "Report for {patient_name} ({patient_gender})."
        formatted_details2 = get_formatted_patient_details("P001", template2)
        print("\nFormatted Details 2:")
        print(formatted_details2)

        template_with_missing_placeholder = "Name: {patient_name}, Unknown: {unknown_field}"
        formatted_details_missing = get_formatted_patient_details("P001", template_with_missing_placeholder)
        print("\nFormatted Details (with unmapped placeholder):")
        print(formatted_details_missing)

    except ValueError as e:
        print(e)

    print("\nAttempting to access a non-existent patient:")
    try:
        template_error = "ID: {patient_id}"
        get_formatted_patient_details("P002", template_error)
    except ValueError as e:
        print(e)
