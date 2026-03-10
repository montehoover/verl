def generate_base_html() -> str:
    """
    Return a basic HTML page with a placeholder for main content.

    Use the token {{MAIN_CONTENT}} as the placeholder you can replace later.
    """
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Welcome</title>
  <style>
    :root { color-scheme: light dark; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; }
    header, footer { padding: 1rem; background: #f5f5f5; color: #111; }
    main { padding: 2rem; }
    @media (prefers-color-scheme: dark) {
      header, footer { background: #1f1f1f; color: #eee; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Welcome</h1>
  </header>
  <main id="content">
    {{MAIN_CONTENT}}
  </main>
  <footer>
    <small>&copy; 2025</small>
  </footer>
</body>
</html>"""
