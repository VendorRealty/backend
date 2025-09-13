def main():
    import os
    import signal
    import sys
    import subprocess
    from pathlib import Path

    project_root = Path(__file__).resolve().parent

    # Configuration (override with env vars if desired)
    host = os.environ.get("HOST", "127.0.0.1")
    port_2d = int(os.environ.get("PORT_2D_TO_3D", "8001"))
    port_materials = int(os.environ.get("PORT_MATERIALS", "8002"))

    py = sys.executable  # default to current interpreter
    # Prefer active virtualenv's python if available (ensures correct site-packages)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_bin = Path(venv) / ("Scripts" if os.name == "nt" else "bin")
        venv_python = venv_bin / ("python.exe" if os.name == "nt" else "python")
        if venv_python.exists():
            py = str(venv_python)

    # Build uvicorn commands
    # 2d-to-3d lives under a non-importable package name; use --app-dir
    cmd_2d = [
        py, "-m", "uvicorn",
        "main:app",
        "--host", host,
        "--port", str(port_2d),
        "--app-dir", str(project_root / "2d-to-3d"),
    ]

    # materials API is a proper package/module
    cmd_materials = [
        py, "-m", "uvicorn",
        "materials.api:app",
        "--host", host,
        "--port", str(port_materials),
    ]

    print("Launching backends:")
    print(f" - 2d-to-3d API: http://{host}:{port_2d} (POST /api/convert)")
    print(f" - Materials API: http://{host}:{port_materials} (POST /api/estimate)")

    procs: list[subprocess.Popen] = []

    def start(cmd: list[str]):
        # Inherit stdout/stderr so logs are visible
        return subprocess.Popen(cmd, cwd=str(project_root))

    try:
        p1 = start(cmd_2d)
        procs.append(p1)
        p2 = start(cmd_materials)
        procs.append(p2)

        # Setup signal handlers for graceful shutdown
        def _shutdown(signum, frame):
            print("\nShutting down child servers...")
            for p in procs:
                if p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass
            # Give them a moment, then kill if needed
            try:
                for p in procs:
                    p.wait(timeout=5)
            except Exception:
                for p in procs:
                    if p.poll() is None:
                        try:
                            p.kill()
                        except Exception:
                            pass
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Wait on children; if any exits, shut down the rest
        import time
        while True:
            exit_codes = [p.poll() for p in procs]
            if any(code is not None for code in exit_codes):
                print("One of the servers exited; shutting down remaining...")
                _shutdown(None, None)
            # Avoid busy loop without relying on signals like SIGCHLD
            time.sleep(0.5)
    except KeyboardInterrupt:
        # Fallback if signal handler didn't catch
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        sys.exit(0)


if __name__ == "__main__":
    main()