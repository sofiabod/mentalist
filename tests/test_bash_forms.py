import pytest

from adapters.mini_swe import _extract_write
from mining.normalize import classify


# Fixture corpus: real bash forms from the captured trajectory's inventory plus
# the common variants the study names. Each row is the verbatim command a model
# emits and the (kind, verb) its real semantics demand.
CLASSIFY_CORPUS = [
    # Redirection writes a file even when the command otherwise only prints.
    ("sed -n '61,75p' fastapi/applications.py", ("read", "free")),
    ("sed -n '61,75p' fastapi/applications.py > temp_init.py", ("edit", "fork")),
    ("sed -n '/def __init__/,/return/p' fastapi/routing.py", ("read", "free")),
    # sed -i is an in-place EDIT.
    ("sed -i 's/foo/bar/' fastapi/routing.py", ("edit", "fork")),
    ("sed -i '/class APIRouter(/,/^    def api_route(/s/x/y/' fastapi/routing.py",
     ("edit", "fork")),
    # cat FILE and cat|sed transforms are READs.
    ("cat fastapi/applications.py", ("read", "free")),
    ("cat temp_init.py | sed 's/    def __init__/    def __init__2/'", ("read", "free")),
    ("nl -ba fastapi/applications.py | grep -A 20 'def __init__'", ("read", "free")),
    # heredoc create behind a compound prefix is an EDIT.
    ("mkdir -p fastapi/middleware && cat > fastapi/middleware/methods.py << 'EOF'\nx = 1\nEOF",
     ("edit", "fork")),
    ("cat > new.py << 'EOF'\nprint(1)\nEOF", ("edit", "fork")),
    # apply_patch is an EDIT.
    ("apply_patch << 'PATCH'\n*** Begin Patch\n*** Add File: a.py\n+hi\n*** End Patch\nPATCH",
     ("edit", "fork")),
    # echo/printf redirection writes are EDITs.
    ("echo 'hello' > note.txt", ("edit", "fork")),
    ("printf 'a\\nb\\n' > out.txt", ("edit", "fork")),
    # append (>>) is non-speculable: a fork cannot faithfully reproduce an append.
    ("echo more >> log.txt", ("append", "never")),
    # searches are grep.
    ("find . -type f -name '*.py' | grep -E 'applications|routing'", ("grep", "free")),
    ("rg 'def api_route' fastapi/", ("grep", "free")),
    ("grep -rn TODO src/", ("grep", "free")),
    # reads.
    ("ls -la", ("read", "free")),
    ("head -20 setup.py", ("read", "free")),
    # test / lint / typecheck via bash.
    ("pytest -q tests/", ("test", "free")),
    ("python -m pytest tests/test_x.py", ("test", "free")),
    ("ruff check .", ("lint", "free")),
    ("mypy src/", ("typecheck", "free")),
    # bare echo (no redirect) and python inline fall through to a read free.
    ("echo hi && ls", ("read", "free")),
]


@pytest.mark.parametrize("command,expected", CLASSIFY_CORPUS)
def test_classify_bash_form(command, expected):
    assert classify("Bash", command) == expected


# Write extraction: verbatim command -> (path, resulting file contents), or None
# for forms that cannot be losslessly parsed from the command string alone.
EXTRACT_CORPUS = [
    ("cat > new.py << 'EOF'\nprint(1)\nx = 2\nEOF", ("new.py", "print(1)\nx = 2\n")),
    ("mkdir -p fastapi/middleware && cat > fastapi/middleware/methods.py << 'EOF'\nx = 1\ny = 2\nEOF",
     ("fastapi/middleware/methods.py", "x = 1\ny = 2\n")),
    ("tee out.txt << EOF\nline\nEOF", ("out.txt", "line\n")),
    ("apply_patch << 'PATCH'\n*** Begin Patch\n*** Add File: a.py\n+hi\n+there\n*** End Patch\nPATCH",
     ("a.py", "hi\nthere\n")),
    ("echo 'hello world' > note.txt", ("note.txt", "hello world\n")),
    ("echo hi > note.txt", ("note.txt", "hi\n")),
    ("printf 'a\\nb\\n' > out.txt", ("out.txt", "a\nb\n")),
    # forms that must return None (unsafe to parse to exact contents).
    ("sed -i 's/foo/bar/' file.py", None),
    ("echo more >> log.txt", None),
    ("echo $HOME > x.txt", None),
    ("printf '%s' \"$x\" > x.txt", None),
    ("cat > a.txt << EOF\nhi\nEOF\ncat > b.txt << EOF\nyo\nEOF", None),
]


@pytest.mark.parametrize("command,expected", EXTRACT_CORPUS)
def test_extract_write_form(command, expected):
    assert _extract_write(command) == expected
