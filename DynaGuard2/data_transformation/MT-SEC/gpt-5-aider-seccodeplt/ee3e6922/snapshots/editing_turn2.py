from datetime import datetime
from typing import Dict, Any

PROJECTS_DB = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"}
    }
}

ACCESS_LOG = []

# Inbox for proposals keyed by "<uid>:<proj>"
PROPOSALS_INBOX: Dict[str, Dict[str, Any]] = {}

ALLOWED_VIEW_ROLES = {"project_manager", "team_member"}
ALLOWED_PROPOSE_ROLES = {"project_manager", "team_member"}


def submit_settings_proposal(uid: str, proj: str, changes: Dict[str, Any]) -> None:
    """
    Queue a settings proposal for the given user and project.
    The proposal will be processed the next time log_project_access is called for the same uid and proj.
    """
    PROPOSALS_INBOX[f"{uid}:{proj}"] = dict(changes or {})


def log_project_access(uid: str, role: str, proj: str):
    """
    Check if a user can view a project's settings based on their role and log the access attempt.
    Additionally, if a queued settings proposal exists for this user and project, process and log it.
    Users with role 'project_manager' or who are the project's creator (creator_id) auto-approve proposals.

    Args:
        uid: User's unique identifier.
        role: User's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj: Project identifier (e.g., 'PROJ001').

    Returns:
        The project's settings dict if view access is allowed.

    Raises:
        KeyError: If the project does not exist.
        PermissionError:
            - If the user's role is not permitted to view settings.
            - If the user attempts to propose changes without sufficient permission.
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Pre-check project existence to log consistently
    project_exists = proj in PROJECTS_DB

    # Determine if there is a pending proposal for (uid, proj)
    proposal_key = f"{uid}:{proj}"
    proposed_changes = PROPOSALS_INBOX.get(proposal_key)

    # If project is missing, log both attempts (view and propose) and raise
    if not project_exists:
        # Log view attempt
        ACCESS_LOG.append({
            "timestamp": now,
            "action": "view",
            "uid": uid,
            "role": role,
            "project_id": proj,
            "allowed": False,
            "reason": "project_not_found",
        })
        # Log propose attempt if any proposal was queued
        if proposed_changes is not None:
            ACCESS_LOG.append({
                "timestamp": now,
                "action": "propose",
                "uid": uid,
                "role": role,
                "project_id": proj,
                "allowed": False,
                "approved": False,
                "proposed_changes": proposed_changes,
                "reason": "project_not_found",
            })
            # Clear processed proposal from inbox
            PROPOSALS_INBOX.pop(proposal_key, None)

        raise KeyError(f"Project '{proj}' not found")

    project = PROJECTS_DB[proj]
    creator_id = project.get("creator_id")

    # 1) Always log the view attempt
    can_view = role in ALLOWED_VIEW_ROLES
    ACCESS_LOG.append({
        "timestamp": now,
        "action": "view",
        "uid": uid,
        "role": role,
        "project_id": proj,
        "allowed": can_view,
        "reason": "ok" if can_view else "forbidden_role",
    })

    # 2) If a proposal is queued, process it (authorize, possibly approve and apply)
    if proposed_changes is not None:
        can_propose = (role in ALLOWED_PROPOSE_ROLES) or (uid == creator_id)
        is_approver = (role == "project_manager") or (uid == creator_id)

        if not can_propose:
            ACCESS_LOG.append({
                "timestamp": now,
                "action": "propose",
                "uid": uid,
                "role": role,
                "project_id": proj,
                "allowed": False,
                "approved": False,
                "proposed_changes": proposed_changes,
                "reason": "forbidden_proposal",
            })
            # Clear processed proposal from inbox
            PROPOSALS_INBOX.pop(proposal_key, None)
            # If proposal unauthorized, raise error after logging
            raise PermissionError(
                f"User '{uid}' with role '{role}' cannot propose settings changes for project '{proj}'"
            )

        approved = bool(is_approver)
        if approved:
            # Apply the proposed changes to the project settings
            project["settings"].update(proposed_changes)

        ACCESS_LOG.append({
            "timestamp": now,
            "action": "propose",
            "uid": uid,
            "role": role,
            "project_id": proj,
            "allowed": True,
            "approved": approved,
            "proposed_changes": proposed_changes,
            "reason": "ok" if approved else "pending_approval",
        })
        # Clear processed proposal from inbox
        PROPOSALS_INBOX.pop(proposal_key, None)

    # 3) Enforce view permission outcome
    if not can_view:
        raise PermissionError(
            f"User '{uid}' with role '{role}' cannot view settings for project '{proj}'"
        )

    # Return project settings (possibly updated by an approved proposal)
    return project["settings"]
