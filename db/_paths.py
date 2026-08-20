"""
db/_paths.py
------------
THE single source of truth for where crisis.db lives on disk.

WHY THIS FILE EXISTS
---------------------
Before this fix, every file in db/ did:

    conn = sqlite3.connect("crisis.db")

That's a RELATIVE path — it resolves against the process's current working
directory, not the project folder. In practice this meant:

  • run_system.py, db/init_db.py, the test suite, etc. all worked fine as
    long as you happened to `cd` into the project root first.
  • streamlit_app.py, on the other hand, already builds an ABSOLUTE path
    (DB_PATH = os.path.join(_ROOT, "crisis.db")) for the one place it reads
    the DB directly (the Stage 5 raw-table preview) — because Streamlit can
    be launched from anywhere (`streamlit run /some/other/dir/streamlit_app.py`,
    a desktop shortcut, a systemd unit with a different WorkingDirectory, a
    Docker CMD, etc).

  Whenever those two didn't agree — i.e. whenever cwd != project root — the
  pipeline would write zone data into a crisis.db file appearing in WHATEVER
  directory the process happened to be launched from, while the UI kept
  reading the (empty/stale) crisis.db sitting next to streamlit_app.py. The
  dashboard would then look exactly like "nothing is happening / the data is
  just whatever was manually typed in" even though the pipeline ran fine.

THE FIX
-------
Compute the path ONCE, relative to this file's own location (which never
moves), and have every db/*.py module + streamlit_app.py import it from here.
"""

import os

# db/_paths.py lives at <project_root>/db/_paths.py, so the project root is
# one directory up from this file — regardless of the caller's cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CRISIS_DB_PATH = os.path.join(PROJECT_ROOT, "crisis.db")
