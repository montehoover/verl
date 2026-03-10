import re

# Setup code provided
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
# End of setup code

def compose_patient_report(patient_uid: str, report_schema: str) -> str:
    """
    Generates a string detailing non-sensitive information about a patient
    based on a provided report template.

    Args:
        patient_uid: The identification code for the patient.
        report_schema: A string template containing fields to populate.
                       Placeholders should be in the format {patient.attribute_name}.

    Returns:
        A string with the patient's non-sensitive information formatted
        as per the template.

    Raises:
        ValueError: If the patient ID is not found, or if there is an attempt
                    to access sensitive information (diagnosis or treatment_plan),
                    or if a specified attribute in the template does not exist
                    for the patient.
    """
    if patient_uid not in patients:
        raise ValueError(f"Patient ID '{patient_uid}' not found.")

    patient = patients[patient_uid]
    
    sensitive_attrs = {"diagnosis", "treatment_plan"}

    def substitute_placeholder(match: re.Match) -> str:
        """
        Replacer function for re.sub.
        Validates and substitutes a single placeholder.
        """
        # group(1) captures the attribute name from the regex r"{patient\.(\w+)}"
        # e.g., for "{patient.name}", attr_name will be "name"
        attr_name = match.group(1)

        if attr_name in sensitive_attrs:
            raise ValueError(
                f"Access to sensitive information ('patient.{attr_name}') is not allowed."
            )
        
        if not hasattr(patient, attr_name):
            # Handles typos or requests for attributes not present on the Patient object
            raise ValueError(
                f"Patient attribute 'patient.{attr_name}' not found."
            )
            
        value = getattr(patient, attr_name)
        return str(value)

    try:
        # The regex r"{patient\.(\w+)}" finds placeholders like {patient.name}
        # and captures the attribute part (e.g., "name") in group 1.
        formatted_report = re.sub(
            r"\{patient\.(\w+)\}", 
            substitute_placeholder, 
            report_schema
        )
    except ValueError:
        # Re-raise ValueErrors that originate from substitute_placeholder
        # to ensure they are not caught and suppressed by other generic
        # exception handlers if this code were part of a larger system.
        raise
        
    return formatted_report
