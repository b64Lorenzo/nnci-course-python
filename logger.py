from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4


_log_file = None


def _one_line(value):
    return (
        str(value).replace("\\", "\\\\").replace("\n", "\\n")
        .replace("\r", "\\r").replace("|", "\\|")
    )


def log(level, component, message):
    global _log_file
    now = datetime.now(timezone.utc)
    if _log_file is None:
        run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_directory = Path("logs") / run_id
        run_directory.mkdir(parents=True)
        _log_file = run_directory / "run.log"

    line = (
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')}|{level.upper()}|"
        f"{_one_line(component)}|{_one_line(message)}"
    )
    print(line, file=sys.stderr)
    with _log_file.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
