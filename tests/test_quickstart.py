"""The quickstart example must actually run — a broken demo is worse than none.

We import examples/quickstart.py by file path (it lives outside the package)
and call its main(), asserting it completes without raising. This guards the
public-API surface the README points newcomers at.
"""
import importlib.util
from pathlib import Path

_QUICKSTART = Path(__file__).resolve().parent.parent / "examples" / "quickstart.py"


def _load_quickstart():
    spec = importlib.util.spec_from_file_location("quickstart", _QUICKSTART)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart_runs(capsys):
    module = _load_quickstart()
    module.main()  # must not raise
    out = capsys.readouterr().out
    # It should have ingested chunks and printed an answer for each question.
    assert "Ingested" in out
    assert out.count("Q:") == 3
    assert "A:" in out
