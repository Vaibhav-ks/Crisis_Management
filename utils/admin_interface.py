# def admin_approval(message):

#     print("\n==============================")
#     print("ADMIN APPROVAL REQUIRED")
#     print(message)

#     while True:

#         decision = input("Approve? (y/n): ").lower()

#         if decision == "y":
#             return True

#         if decision == "n":
#             return False

#         print("Enter y or n")
"""
utils/admin_interface.py
------------------------
Admin approval helpers for the crisis pipeline.

"""

import time


# ── Terminal / blocking version ───────────────────────────────────────────────

def admin_approval(prompt: str = "Approve?", timeout_s: int = 0) -> bool:
    """
    Ask the operator via the terminal.

    Parameters
    ----------
    prompt    : Question to display
    timeout_s : If > 0, automatically approve after this many seconds
                (useful for automated tests).  0 = wait forever.

    Returns
    -------
    bool — True if approved, False if rejected
    """
    if timeout_s > 0:
        print(f"\n[ADMIN] {prompt} (auto-approving in {timeout_s}s)")
        time.sleep(timeout_s)
        print("[ADMIN] Auto-approved ✓")
        return True

    while True:
        print(f"\n[ADMIN] {prompt}")
        answer = input("  Enter 'y' to approve, 'n' to reject: ").strip().lower()
        if answer in ("y", "yes"):
            print("[ADMIN] Approved ✓")
            return True
        elif answer in ("n", "no"):
            print("[ADMIN] Rejected ✗")
            return False
        else:
            print("  Please enter 'y' or 'n'.")


# ── Streamlit / non-blocking version ─────────────────────────────────────────

def admin_approval_streamlit(key: str):
    """
    Non-blocking admin approval for Streamlit.

    Usage in a stage function:
        decision = admin_approval_streamlit("resource_approval")
        if decision is None:
            return       # waiting for button click
        elif decision:
            proceed_to_next_stage()
        else:
            go_back_and_redo()

    Parameters
    ----------
    key : str
        Base key for st.session_state.  Two sub-keys are used internally:
            f"{key}_decision"   — stores True / False / None
            f"{key}_rendered"   — prevents duplicate button render

    Returns
    -------
    None   — no decision yet
    True   — operator approved
    False  — operator rejected
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError(
            "streamlit is not installed.  "
            "Use admin_approval() for terminal mode."
        )

    decision_key = f"{key}_decision"

    # Return stored decision if already made
    if decision_key in st.session_state:
        return st.session_state[decision_key]

    # Render approve / reject buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", key=f"{key}_btn_approve"):
            st.session_state[decision_key] = True
            st.rerun()
    with col2:
        if st.button("❌ Reject", key=f"{key}_btn_reject"):
            st.session_state[decision_key] = False
            st.rerun()

    return None


def reset_approval(key: str):
    """
    Clear a previous decision so the operator can decide again.
    Call this when re-running a stage after a rejection.
    """
    try:
        import streamlit as st
        for suffix in ("_decision", "_btn_approve", "_btn_reject"):
            st.session_state.pop(f"{key}{suffix}", None)
    except ImportError:
        pass