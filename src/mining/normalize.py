import re
import shlex

from sfx.schema import ToolEvent, Outcome

READ_TOOLS = {"Read", "WebFetch", "view_image", "read_mcp_resource", "NotebookRead"}
GREP_TOOLS = {"Grep", "Glob", "WebSearch", "ToolSearch"}
EDIT_TOOLS = {"Edit", "apply_patch", "NotebookEdit", "Write"}
SUBLLM_TOOLS = {"Agent", "Task", "TaskCreate", "TaskOutput", "Skill"}
BASH_TOOLS = {"Bash", "shell_command", "shell", "exec_command", "write_stdin"}

CMD_KINDS = [
    (("pytest", "unittest", "jest", "go test", "cargo test", "npm test", "vitest"), ("test", "free")),
    (("ruff", "flake8", "eslint", "pylint", "clippy", "black --check"), ("lint", "free")),
    (("mypy", "tsc", "pyright", "py_compile", "pyre"), ("typecheck", "free")),
    (("make", "cargo build", "npm run build", "tsc --build", "go build", "cmake"), ("build", "free")),
    (("rg", "grep", "find", "fd", "ag", "ls", "glob"), None),
    (("cat", "head", "tail", "nl", "less", "stat", "pwd", "pdftotext", "pdfinfo"), ("read", "free")),
    (("sed", "awk", "mv", "cp", "mkdir", "touch", "rm", "chmod", "tee"), ("edit", "fork")),
    (("echo", "python", "node", "bash", "sh"), None),
]


WRITE_CMDS = ("rm", "mkdir", "mv", "cp", "chmod", "touch", "tee")


def _is_bash_write(s):
    if "apply_patch" in s:
        return True
    if re.search(r"<<-?\s*['\"]?\w+", s):
        return True
    unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", "", s)
    # stderr redirects and redirects after read commands also create/truncate files.
    return bool(re.search(r">{1,2}\s*\S", unquoted))


def _is_inplace(s):
    if not re.search(r"\b(g?sed|perl)\b", s):
        return False
    try:
        words = shlex.split(s)
    except ValueError:
        return False
    return any(word.startswith("--in-place") or re.match(r"^-[a-zA-Z]*i", word)
               for word in words[1:])


READ_LEADERS = ("sed", "cat", "head", "tail", "nl", "less", "stat", "grep", "rg", "find", "fd", "ag", "ls")


def _segment_is_write(seg):
    seg = seg.strip()
    if _is_inplace(seg) or _is_bash_write(seg):
        return True
    try:
        words = shlex.split(seg)
    except ValueError:
        return False
    # Check mutating variants before assigning a semantic test/lint/read kind.
    if any(word.split("=", 1)[0] in {
        "--fix", "--fix-only", "--unsafe-fixes", "--write", "-w", "-o",
        "--output", "--output-file", "--output-dir", "--out", "--outDir",
        "--outFile", "--build", "--incremental", "--junitxml", "--junit-xml",
        "--cache-clear", "--cov-report", "--updateSnapshot", "-u",
    } for word in words[1:]):
        return True
    if words and words[0] in {"ruff", "black", "prettier"}:
        if "format" in words and "--check" not in words and "--diff" not in words:
            return True
        if words[0] == "black" and "--check" not in words and "--diff" not in words:
            return True
    lead = re.match(r"([\w./-]+)", seg)
    tok = lead.group(1).rsplit("/", 1)[-1] if lead else ""
    if tok in WRITE_CMDS:
        return True
    if re.search(r"\bxargs\b.*\b(" + "|".join(WRITE_CMDS) + r")\b", seg):
        return True
    if tok in READ_LEADERS:
        return False
    return _is_bash_write(seg)


_INSTALL_RE = re.compile(
    r"\b(pip3?\s+install|pip3?\s+download|python3?\s+-m\s+pip\s+install|"
    r"npm\s+install|npm\s+i\b|npm\s+ci|yarn\s+add|pnpm\s+add|"
    r"cargo\s+add|poetry\s+add|"
    r"apt(-get)?\s+install|apk\s+add|dnf\s+install|yum\s+install|brew\s+install)")
_NETWORK_RE = re.compile(
    r"(^|[|&;]\s*)(curl|wget|nc|scp|rsync|ssh|sftp|ftp)\b|https?://|ftp://")


def _is_append(s):
    if re.search(r"\btee\s+(?:-a\b|--append\b)", s):
        return True
    unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", "", s)
    return bool(re.search(r"(?<![|&2])>>", unquoted))


DESTRUCTIVE_CMDS = ("rm", "dd", "mkfs", "shred", "truncate")
_UNSAFE_META_RE = re.compile(r";|\$\(|`|[<>]\(|(?<![&>])&(?!&)")
_DESTRUCTIVE_FLAG_RE = re.compile(r"(?:^|\s)(?:-delete|--delete)\b")


def _is_never_command(s):
    if _UNSAFE_META_RE.search(s) or _DESTRUCTIVE_FLAG_RE.search(s):
        return True
    # These search options execute arbitrary programs or write output files.
    if re.search(r"\bfind\b.*\s-(?:exec(?:dir)?|ok(?:dir)?|fprint(?:0|f)?|fls)\b", s):
        return True
    if re.search(r"\bfd\b.*\s(?:-[xX]|--exec(?:-batch)?)(?:\s|=|$)", s):
        return True
    if re.search(r"\brg\b.*\s--(?:pre|hostname-bin)(?:\s|=|$)", s):
        return True
    segs = re.split(r"&&|\|\||;|\|", s)
    for i, seg in enumerate(segs):
        lead = re.match(r"(?:\w+=\S+\s+)*([\w./-]+)", seg.strip())
        tok = lead.group(1).rsplit("/", 1)[-1] if lead else ""
        if i > 0 and tok in DESTRUCTIVE_CMDS:
            return True
        if re.search(r"\bxargs\b", seg) and any(re.search(rf"\b{d}\b", seg) for d in DESTRUCTIVE_CMDS):
            return True
    return False


def _command_segments(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    segments, words = [], []
    for token in lexer:
        # shlex groups adjacent punctuation, e.g. an operator followed by a
        # newline or multiple blank lines. All of it still separates commands.
        if token and set(token) <= set(";&|\n"):
            segments.append(shlex.join(words))
            words = []
        else:
            words.append(token)
    segments.append(shlex.join(words))
    return segments


def _classify_command(skel):
    s = skel.strip()
    if s.startswith("git") or " git " in f" {s} " or "&& git" in s:
        return ("git", "never")
    if _is_never_command(s):
        return ("unsafe", "never")
    # non-speculable mutations: order/remote/package state a fork cannot reproduce
    if _INSTALL_RE.search(s):
        return ("install", "never")
    if _NETWORK_RE.search(s):
        return ("network", "never")
    if _is_append(s):
        return ("append", "never")
    # A read/test at the start must never hide writes later in the shell command.
    # Heredoc writes are stream-only; they need not be parsed as executable lines.
    if _is_bash_write(s):
        return ("edit", "fork")
    try:
        segs = _command_segments(s)
    except ValueError:
        return ("unsafe", "never")
    if any(_segment_is_write(seg) for seg in segs):
        return ("edit", "fork")
    if len(segs) > 1:
        # Classify every command, including an unknown command after a safe one.
        # Quoted separators can conservatively reject speculation.
        classified = [_classify_command(seg) for seg in segs if seg.strip()]
        if not classified or any(verb != "free" for _, verb in classified):
            return ("unsafe", "never")
        return classified[0]
    if re.match(r"(?:python3?\s+-m\s+)?(?:pytest|unittest)(?:\s|$)", s):
        return ("test", "free")
    if re.match(r"(python3?|node|ruby)\s+(?!-)[\w./-]+\.(py|js|rb)$", s) \
       or re.match(r"go\s+run\s+[\w./-]+\.go$", s):
        if "manage.py" not in s:
            return ("run", "free")
    # Arbitrary interpreter snippets/modules/scripts must not inherit read/free
    # merely because their executable starts with a familiar interpreter name.
    if re.match(r"(?:python3?|node|ruby|bash|sh|zsh)\s+", s):
        return ("unknown", "never")
    if re.match(r"(?:\S*/)?sed(?:\s|$)", s):
        return ("read", "free")
    for prefixes, kind in CMD_KINDS:
        for p in prefixes:
            if s == p or s.startswith(p + " "):
                if kind is None:
                    if p in ("rg", "grep", "find", "fd", "ag", "ls", "glob"):
                        return ("grep", "free") if p in ("rg", "grep", "find", "fd", "ag") else ("read", "free")
                    return ("read", "free")
                return kind
    return ("unknown", "never")


def classify(tool_name, command_skeleton=None):
    if tool_name in READ_TOOLS:
        return ("read", "free")
    if tool_name in GREP_TOOLS:
        return ("grep", "free")
    if tool_name in EDIT_TOOLS:
        return ("edit", "fork")
    if tool_name in SUBLLM_TOOLS:
        return ("sub-LLM", "free")
    if tool_name in BASH_TOOLS and command_skeleton:
        return _classify_command(command_skeleton)
    return ("unknown", "never")


def _status(is_error, kind):
    if kind in ("test", "lint", "typecheck", "build"):
        return "FAIL" if is_error else "PASS"
    return "ERR" if is_error else "OK"


def _latency(tool):
    lat = tool.get("tool_internal_latency_ms")
    if lat is None:
        lat = tool.get("tool_wall_latency_ms")
    return lat


def normalize_row(row):
    tools ={t["tool_call_id"]: t for t in row.get("tools") or [] if t.get("tool_call_id")}
    events = []
    order = 0
    for ev in row.get("timing_events") or []:
        if ev.get("event_type") != "tool_call":
            continue
        cid = ev.get("tool_call_id")
        tool = tools.get(cid, {})
        name = ev.get("tool_name") or tool.get("tool_name")
        if not name:
            continue
        kind, verb = classify(name, tool.get("command_skeleton"))
        is_error = bool(tool.get("is_error"))
        status = _status(is_error, kind)
        events.append(
            ToolEvent(
                t=float(row["round_index"]) + order / 1000.0,
                kind=kind,
                verb=verb,
                role="main",
                epoch=0,
                args={"latency_ms": _latency(tool)},
                outcome=Outcome(kind, status),
            )
        )
        order += 1
    return events
