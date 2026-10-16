"""Docker isolation for untrusted candidate commands.

Only the candidate workspace is mounted by default. A trusted controller may
explicitly provide an ephemeral output directory, mounted at /output. Never put
contracts, reference implementations, hidden answers or evidence in either
mount. Docker shares the host kernel; this is an engineering boundary, not a
formal security guarantee against kernel vulnerabilities or hardware channels.
The file-size limit is per file; writable bind mounts have no aggregate disk
quota. Captured output is separately bounded by WorkspaceToolSuite/the controller.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import uuid


class DockerSandbox:
    """Build one owned container command and remove it after execution.

    ``build_command`` returns argv, suitable for ``Popen(shell=False)``. Always
    call ``cleanup`` in a finally block, including when the client is terminated.
    Use one instance per command. The image must already exist locally; runtime
    commands never pull images or fall back to host execution.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        image: str = "antigravity-orion:local",
        *,
        output_directory: str | Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve(strict=True)
        if not self.workspace_root.is_dir():
            raise ValueError("Workspace must be a directory")
        self.uid = os.getuid()
        self.gid = os.getgid()
        if self.uid == 0:
            raise ValueError("Docker sandbox refuses to run as root; use a non-root host user")
        if not image or image.startswith("-") or any(c.isspace() for c in image):
            raise ValueError("Invalid sandbox image")
        self.image = image
        self.container_name = "antigravity-sandbox-" + uuid.uuid4().hex
        self.output_directory = None
        if output_directory is not None:
            output = Path(output_directory).resolve(strict=True)
            if not output.is_dir():
                raise ValueError("Output must be an existing directory")
            if output.is_relative_to(self.workspace_root) or self.workspace_root.is_relative_to(output):
                raise ValueError("Output and candidate workspace must not overlap")
            self.output_directory = output
        for path in (self.workspace_root, self.output_directory):
            if path is not None and any(char in str(path) for char in (",", "\n", "\r")):
                raise ValueError("Docker mount paths cannot contain commas or newlines")

    @staticmethod
    def client_environment() -> dict[str, str]:
        """Use the local Docker socket without host config, contexts or secrets.

        These values configure the host Docker client only. The container gets
        just the explicit --env values in ``build_command`` and image defaults.
        """
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/nonexistent"}

    def build_command(
        self, command: str, cwd_relative: str = ".", read_only: bool = False
    ) -> list[str]:
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")
        relative = Path(cwd_relative)
        if relative.is_absolute():
            raise ValueError("cwd_relative must be relative to candidate workspace")
        cwd = (self.workspace_root / relative).resolve(strict=True)
        if not cwd.is_relative_to(self.workspace_root):
            raise PermissionError("Working directory escapes candidate workspace")
        if not cwd.is_dir():
            raise ValueError("Working directory must be a directory")
        container_cwd = "/workspace"
        if cwd != self.workspace_root:
            container_cwd += "/" + cwd.relative_to(self.workspace_root).as_posix()
        mount = f"type=bind,src={self.workspace_root},dst=/workspace,bind-recursive=disabled"
        if read_only:
            mount += ",readonly"
        args = [
            "docker", "run", "--rm", "--pull=never", "--name", self.container_name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--user", f"{self.uid}:{self.gid}",
            "--cpus=2", "--memory=4g", "--memory-swap=4g", "--pids-limit=256",
            "--ulimit=nofile=1024:1024", "--ulimit=fsize=16777216:16777216",
            "--log-driver=none", "--init", "--stop-timeout=1",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
            "--shm-size=64m", "--mount", mount, "--workdir", container_cwd,
            "--env", "HOME=/tmp", "--env", "PYTHONUNBUFFERED=1",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=1",
            "--env", "OPENBLAS_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
        ]
        if self.output_directory is not None:
            args += ["--mount", f"type=bind,src={self.output_directory},dst=/output,bind-recursive=disabled"]
        return args + [self.image, "/bin/bash", "--noprofile", "--norc", "-c", command]

    def cleanup(self) -> None:
        """Force-remove this container, killing every process in its PID namespace.

        Missing containers are normal with --rm. Daemon/permission failures are
        surfaced so a caller cannot report successful isolation or cleanup.
        """
        try:
            result = subprocess.run(
                ["docker", "rm", "--force", self.container_name],
                env=self.client_environment(), capture_output=True, text=True, timeout=10,
            )
        except FileNotFoundError:
            # The command could not start either: there is no host fallback.
            return
        if result.returncode and "No such container" not in result.stderr:
            raise RuntimeError(f"Docker sandbox cleanup failed: {result.stderr.strip()}")
