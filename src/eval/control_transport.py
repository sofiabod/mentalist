"""Serialized persistent control over a locally accessible daemon socket.

This avoids a Python process per daemon operation, without bypassing the daemon
protocol. Socket RPC also works through the sidecar's private host bind mount:
repository and scratch arguments remain in the daemon's namespace. Snapshot
dispatch is different: it reads the caller's filesystem directly. The sidecar
therefore keeps snapshots in its existing container execution path rather than
passing them through this host-side client.
"""
import asyncio
import os

from adapters.protocol import Client
from eval.sfx_client_cli import _checked, dispatch


class InSandboxControlTransport:
    def __init__(self, socket_path):
        self.socket_path = os.fspath(socket_path)
        if not os.path.isabs(self.socket_path):
            raise ValueError("in-sandbox control requires an absolute socket path")
        self._client = None
        self._lock = asyncio.Lock()
        self._closed = False

    def _discard_client(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except OSError:
                pass

    def _request(self, argv):
        try:
            if argv[0] != "snapshot" and self._client is None:
                self._client = Client(self.socket_path)
            return _checked(dispatch(self._client, argv))
        except BaseException:
            self._discard_client()
            raise

    async def request(self, *argv):
        if len(argv) < 2:
            raise ValueError("sfx control requires an operation and session")
        args = tuple(str(value) for value in argv)
        async with self._lock:
            if self._closed:
                raise RuntimeError("in-sandbox control transport is closed")
            task = asyncio.create_task(asyncio.to_thread(self._request, args))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        pass
                    except BaseException:
                        break
                if not task.cancelled():
                    task.exception()
                self._discard_client()
                raise

    async def aclose(self):
        async with self._lock:
            self._closed = True
            self._discard_client()

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("in-sandbox control transport is closed")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()
