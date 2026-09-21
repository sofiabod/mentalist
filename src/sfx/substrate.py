from pathlib import Path

from sfx.fork import WritePathError, choose_substrate, fork, validate_write_path


class FilesystemSubstrate:
    def __init__(self, apply_write, run_in_fork, validate_handle=None):
        self._apply_write = apply_write
        self._run_in_fork = run_in_fork
        self.validate_handle = validate_handle

    def probe(self, repo, scratch):
        """Cheap capability check (no clone); raises ForkError if no COW substrate."""
        choose_substrate(Path(scratch).resolve(strict=True), Path(repo).resolve(strict=True))

    def fork(self, repo, scratch):
        return fork(repo, scratch)

    def apply(self, handle, write):
        if "path" not in write.args:
            raise WritePathError("streamed write missing 'path'")
        validate_write_path(write.args["path"], handle.path)
        # A cloned inode can have fewer aliases than its authoritative source,
        # especially when a hardlink lives outside the repository. An ordinary
        # authoritative redirect would edit every alias; our atomic replacement
        # would not. Reject that unsupported write before any callback runs.
        if getattr(handle, "repo", None) is None:
            raise WritePathError("filesystem fork is missing its source root")
        validate_write_path(write.args["path"], handle.repo)
        self._apply_write(handle.path, write)

    def run_get(self, handle, hop):
        if self.validate_handle is not None:
            self.validate_handle(handle, hop)
        return self._run_in_fork(handle.path, hop)
