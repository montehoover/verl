def format_string(template: str, content: str) -> str:
    """
    Replaces a placeholder in the template with the provided content.

    Args:
        template: The string template with a placeholder.
                  It is assumed the placeholder is "{content}".
        content: The string content to insert into the template.

    Returns:
        The formatted string with the placeholder replaced by content.
    """
    return template.replace("{content}", content)

if __name__ == '__main__':
    # Example usage:
    template_string = "Hello, {content}! Welcome to our service."
    user_name = "Alice"
    formatted_greeting = format_string(template_string, user_name)
    print(formatted_greeting)

    template_string_2 = "The weather today is {content}."
    weather_condition = "sunny"
    formatted_weather = format_string(template_string_2, weather_condition)
    print(formatted_weather)
