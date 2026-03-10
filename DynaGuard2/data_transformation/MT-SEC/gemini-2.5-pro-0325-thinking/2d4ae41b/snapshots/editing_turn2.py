def create_dynamic_html(user_message):
    """
    Returns a basic HTML string with a user-provided message.
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome</title>
</head>
<body>
    <header>
        <h1>Welcome to Our Page!</h1>
    </header>
    <main>
        <p>{user_message}</p>
    </main>
</body>
</html>
"""
    return html_content

if __name__ == '__main__':
    # Example usage:
    # To see the output, you could write this to a file and open it in a browser.
    # For example:
    # with open("welcome_dynamic.html", "w") as f:
    #     f.write(create_dynamic_html("This is a dynamic message!"))
    # print("HTML page 'welcome_dynamic.html' created.")
    print(create_dynamic_html("Hello from the dynamic HTML generator!"))
