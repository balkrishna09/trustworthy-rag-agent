"""
Collaboration test file for Trustworthy RAG project.
Both collaborators can edit their respective sections without merge conflicts.
"""

# ─────────────────────────────────────────────
# Collaborator 1 (Balkrishna) — edit below
# ─────────────────────────────────────────────

COLLABORATOR_1 = "Balkrishna"
COLLABORATOR_1_NOTE = "Setup complete, FARMI endpoint configured."

# ─────────────────────────────────────────────
# Collaborator 2 (Friend) — edit below
# ─────────────────────────────────────────────

COLLABORATOR_2 = ""          # Add your name here
COLLABORATOR_2_NOTE = ""     # Add a short note here

# ─────────────────────────────────────────────


def greet(name: str, note: str) -> str:
    """Return a greeting for a collaborator.

    Args:
        name: Collaborator name.
        note: Short status note.

    Returns:
        Formatted greeting string.
    """
    if not name:
        return "Collaborator slot is empty — fill in your name above."
    return f"Hello from {name}! Note: {note}"


if __name__ == "__main__":
    print(greet(COLLABORATOR_1, COLLABORATOR_1_NOTE))
    print(greet(COLLABORATOR_2, COLLABORATOR_2_NOTE))
