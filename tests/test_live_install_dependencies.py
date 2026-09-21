import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
from harbor.agents.installed.base import BaseInstalledAgent

from eval.sfx_live_agent import SFXLiveAgent


class InstallEnvironment:
    default_user = "agent"

    def __init__(self, *, proot=False, apt=True, install_code=0, postcheck=True):
        self.proot = proot
        self.apt = apt
        self.install_code = install_code
        self.postcheck = postcheck
        self.calls = []
        self.uploads = []

    async def exec(self, *, command, **kwargs):
        self.calls.append((command, kwargs))
        if command == "command -v proot >/dev/null 2>&1":
            code = 0 if self.proot else 1
        elif command == "command -v apt-get >/dev/null 2>&1":
            code = 0 if self.apt else 1
        elif command.startswith("apt-get update &&"):
            code = self.install_code
            self.proot = self.postcheck and code == 0
        else:
            code = 0
        return SimpleNamespace(return_code=code, stdout="", stderr="")

    async def upload_dir(self, *paths):
        self.uploads.append(paths)

    async def upload_file(self, *paths):
        self.uploads.append(paths)


def agent(tmp_path, monkeypatch, view="proot"):
    instance = SFXLiveAgent(tmp_path / "logs", fork_path_view=view)
    async def no_daemon(*args):
        return None
    monkeypatch.setattr(instance, "_launch_daemon", no_daemon)
    assert instance.ensure_system_dependencies.__func__ is BaseInstalledAgent.ensure_system_dependencies
    return instance


def test_real_harbor_dependency_allowlist_rejects_proot(tmp_path):
    instance = SFXLiveAgent(tmp_path / "logs")
    with pytest.raises(ValueError, match="Unknown system dependencies: proot"):
        asyncio.run(instance.ensure_system_dependencies(InstallEnvironment(), ("proot",)))


def test_full_install_uses_actual_harbor_helper_and_explicit_proot_install(tmp_path, monkeypatch):
    environment = InstallEnvironment()
    asyncio.run(agent(tmp_path, monkeypatch).install(environment))
    command, _ = environment.calls[0]
    assert "command -v git" in command and "command -v python3" in command
    installs = [(command, kw) for command, kw in environment.calls if command.startswith("apt-get update")]
    assert installs == [("apt-get update && apt-get install -y --no-install-recommends proot",
                         {"user": "root", "env": {"DEBIAN_FRONTEND": "noninteractive"}, "timeout_sec": 120})]
    assert environment.proot and len(environment.uploads) == 2
    probes = [kw for command, kw in environment.calls if command == "command -v proot >/dev/null 2>&1"]
    assert len(probes) == 2 and all(kw["user"] == "agent" for kw in probes)


@pytest.mark.parametrize("view,preinstalled", [("cwd", False), ("proot", True)])
def test_no_package_install_for_cwd_or_preinstalled_binary(tmp_path, monkeypatch, view, preinstalled):
    environment = InstallEnvironment(proot=preinstalled)
    asyncio.run(agent(tmp_path, monkeypatch, view).install(environment))
    assert not any(command.startswith("apt-get") for command, _ in environment.calls)
    assert len(environment.uploads) == 2


@pytest.mark.parametrize("options,message", [
    ({"apt": False}, "preinstalled proot"),
    ({"install_code": 100}, "exit code 100"),
    ({"postcheck": False}, "without an executable"),
])
def test_missing_proot_fails_before_upload_or_daemon_start(tmp_path, monkeypatch, options, message):
    environment = InstallEnvironment(**options)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(agent(tmp_path, monkeypatch).install(environment))
    assert environment.uploads == []
