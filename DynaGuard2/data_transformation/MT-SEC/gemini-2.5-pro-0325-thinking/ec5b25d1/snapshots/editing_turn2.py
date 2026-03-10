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

def print_patient_details(patient_code, format_template):
    """
    Formats patient details (name, age, gender) for a given patient code
    using a provided template.

    Args:
        patient_code (str): The ID of the patient.
        format_template (str): A string template with placeholders like
                               '{patient_name}', '{patient_age}', '{patient_gender}'.

    Returns:
        str: The formatted string with patient details.

    Raises:
        ValueError: If the patient_code is not found in the patients dictionary.
    """
    if patient_code not in patients:
        raise ValueError(f"Patient ID '{patient_code}' not found.")
    
    patient = patients[patient_code]
    
    # Prepare a dictionary with available data for formatting
    # This allows str.format_map to ignore missing keys in the template
    # or raise KeyError if a key in the template is not in data_map,
    # depending on how format_map is used or if a custom SafeFormatter is used.
    # For simplicity, we'll provide the known fields.
    data_map = {
        'patient_code': patient.id,
        'patient_name': patient.name,
        'patient_age': patient.age,
        'patient_gender': patient.gender
        # Sensitive fields like diagnosis and treatment_plan are intentionally omitted
    }
    
    # Using str.format_map to handle placeholders.
    # If a placeholder in the template is not in data_map, it will raise a KeyError.
    # To handle missing placeholders gracefully (e.g., leave them as is or replace with a default),
    # a more complex approach or a custom Formatter subclass would be needed.
    # For now, we assume the template will only use the provided keys.
    # A simple way to make it "graceful" for missing keys in the template
    # (i.e., if template has {unknown_field}) is to use a defaultdict
    # or a custom class that returns an empty string for missing keys.
    
    # For this implementation, we will rely on the template using known placeholders.
    # If a placeholder like {diagnosis} was in the template, it would cause an error
    # unless 'diagnosis' was added to data_map.
    
    # A common way to handle potentially missing keys in the template gracefully
    # is to use a dictionary that returns a placeholder string for missing keys.
    class SafeDict(dict):
        def __missing__(self, key):
            return f'{{{key}}}' # Or return an empty string: ''

    safe_data_map = SafeDict(data_map)
    return format_template.format_map(safe_data_map)

if __name__ == '__main__':
    # Example usage:
    template1 = "Patient Report:\nID: {patient_code}\nName: {patient_name}\nAge: {patient_age}\nGender: {patient_gender}"
    template2 = "Name: {patient_name}, Age: {patient_age}, Diagnosis: {patient_diagnosis}" # diagnosis is not in data_map

    try:
        report1 = print_patient_details("P001", template1)
        print(report1)
    except ValueError as e:
        print(e)

    print("\n--- Attempting with a non-existent patient ID ---")
    try:
        report_error = print_patient_details("P002", template1) # This will raise a ValueError
        print(report_error)
    except ValueError as e:
        print(e)

    print("\n--- Attempting with a template that has an unfillable placeholder ---")
    try:
        report2 = print_patient_details("P001", template2)
        print(report2) # {patient_diagnosis} will remain as is due to SafeDict
    except ValueError as e: # Should not happen for P001
        print(e)
