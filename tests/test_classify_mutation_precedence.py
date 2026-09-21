import pytest

from mining.normalize import classify


@pytest.mark.parametrize("command", [
    "pytest --version && touch unexpected.txt",
    "pytest --version || touch unexpected.txt",
    "pytest --version\ntouch unexpected.txt",
    "pytest --version\n\ntouch unexpected.txt",
    "pytest --version &&\ntouch unexpected.txt",
    "pytest --version | tee unexpected.txt",
    "pytest --version > unexpected.txt",
    "pytest --version 2> unexpected.txt",
    "cat input.txt > unexpected.txt",
    "sed -n '1,4p' input.txt > unexpected.txt",
    "ruff check . --fix",
    "ruff check . --fix-only",
    "ruff format .",
    "eslint --fix .",
    "prettier --write .",
    "pytest --junitxml=report.xml",
    "sed -ni '1p' input.txt",
    "sed -i.bak 's/a/b/' input.txt",
])
def test_mutating_forms_are_never_free(command):
    assert classify("Bash", command)[1] != "free"


@pytest.mark.parametrize("command", [
    "pytest --version && unrecognized_program",
    "cat file | unrecognized_program",
    "pytest --version & touch unexpected.txt",
    "find . -exec touch unexpected.txt +",
    "fd -x touch unexpected.txt",
    "pytest-custom-command",
    "lswrite unexpected.txt",
    "catastrophe unexpected.txt",
    "rgmutate unexpected.txt",
    "sedmutate unexpected.txt",
    "python -c 'open(\"unexpected.txt\", \"w\").close()'",
    "python -m arbitrary_module",
    "bash -c 'touch unexpected.txt'",
    "sh arbitrary_script.sh",
    "node -e 'require(\"fs\").writeFileSync(\"unexpected.txt\", \"x\")'",
    "rg --pre='touch unexpected.txt' pattern",
])
def test_unknown_executables_and_effectful_compounds_rejected(command):
    assert classify("Bash", command)[1] == "never"


def test_mentions_do_not_change_command_kind():
    assert classify("Bash", "cat pytest.py") == ("read", "free")
    assert classify("Bash", "echo pytest") == ("read", "free")


def test_readonly_pipeline_with_quoted_pattern_stays_free():
    assert classify("Bash", "find . -type f | grep -E 'alpha|beta'") == ("grep", "free")
