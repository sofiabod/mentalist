"""Container integration tests are explicit opt-in; normal pytest is CPU-only."""

import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-docker", action="store_true", default=False,
                     help="Run Docker integration tests (may build/pull benchmark images).")


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: starts real Docker containers; requires --run-docker")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-docker"):
        return
    skip = pytest.mark.skip(reason="Docker integration tests require --run-docker")
    for item in items:
        if item.get_closest_marker("docker"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def docker_available(pytestconfig):
    if not pytestconfig.getoption("--run-docker"):
        pytest.skip("Docker integration tests require --run-docker")
    try:
        result = subprocess.run(["docker", "version"], capture_output=True, timeout=3)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("docker not available")
    if result.returncode != 0:
        pytest.skip("docker not available")
