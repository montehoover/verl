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

ALLOWED_ATTRIBUTES = {'id', 'name', 'age', 'gender'}
RESTRICTED_ATTRIBUTES = {'diagnosis', 'treatment_plan'}


def _fill_template_data(report_template: str, patient: Patient, allowed_attrs: set, restricted_attrs: set) -> dict:
    """
    Parses a report template and creates a data dictionary for filling placeholders.

    Args:
        report_template: The string template with placeholders.
        patient: The Patient object containing the data.
        allowed_attrs: A set of attribute names that are allowed to be accessed.
        restricted_attrs: A set of attribute names that are restricted.

    Returns:
        A dictionary where keys are placeholders from the template and values are
        the corresponding patient data.

    Raises:
        ValueError: If a placeholder is malformed, refers to a restricted attribute,
                    or refers to an unknown/non-whitelisted attribute.
    """
    data_for_template = {}
    formatter = string.Formatter()

    for literal_text, field_name, format_spec, conversion in formatter.parse(report_template):
        if field_name:
            if not field_name.startswith("patient."):
                raise ValueError(f"Invalid placeholder format: '{field_name}'. Expected 'patient.attribute'.")
            
            parts = field_name.split('.', 1)
            if len(parts) < 2 or not parts[1]: # Ensure there is an attribute name after "patient."
                raise ValueError(f"Invalid placeholder format: '{field_name}'. Attribute name missing.")

            attr_name = parts[1]

            if attr_name in restricted_attrs:
                raise ValueError(f"Attempt to access restricted medical data: '{attr_name}' in placeholder '{field_name}'")
            
            if attr_name in allowed_attrs:
                if not hasattr(patient, attr_name):
                    # This case should ideally not happen if Patient class and allowed_attrs are consistent
                    raise ValueError(f"Internal configuration error: Attribute '{attr_name}' allowed but not found on Patient object.")
                data_for_template[field_name] = getattr(patient, attr_name)
            else:
                raise ValueError(f"Unknown or not whitelisted patient attribute: '{attr_name}' in placeholder '{field_name}'")
    return data_for_template


def create_patient_report(id: str, report_template: str) -> str:
    """
    Generates a patient report by filling a template with patient information.

    Args:
        id: The unique identifier of the patient.
        report_template: The string template with placeholders like '{patient.attribute}'.

    Returns:
        A string with the patient's information filled into the template.

    Raises:
        ValueError: If the patient ID is invalid, if the template attempts to access
                    restricted medical data, or if a placeholder is malformed or
                    refers to an unknown attribute.
    """
    if id not in patients:
        raise ValueError(f"Invalid patient ID: {id}")

    patient = patients[id]
    
    try:
        data_for_template = _fill_template_data(report_template, patient, ALLOWED_ATTRIBUTES, RESTRICTED_ATTRIBUTES)
        return report_template.format_map(data_for_template)
    except KeyError as e:
        # This might happen if format_map encounters a placeholder not processed by _fill_template_data,
        # or if the template has syntax errors not caught by string.Formatter.parse (e.g. unmatched braces not part of a field)
        raise ValueError(f"Error formatting report. Invalid placeholder or template structure: {e}")
