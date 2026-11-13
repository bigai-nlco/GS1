"""Supervise native GS1 HTTP replicas on explicitly selected GPUs."""

import argparse
import asyncio
import ctypes
import fcntl
import json
import os
import signal
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

from gs1.api.options import FIELDS, FLAGS, add_serving_arguments, serving_settings


def serving_argv(settings):
    values = settings.model_dump()
    result = ["--model-path", str(settings.artifact), "--model-name", settings.local_model]
    for name in (
        *FIELDS,
        "engine",
        "device",
        "precision",
        "runtime_profile",
        "graph_warmup_requests",
        "graph_capture_policy",
        "graph_slots",
    ):
        if values[name] is not None:
            result += ["--" + name.replace("_", "-"), str(values[name])]
    result += [flag for name, flag in FLAGS.items() if values[name]]
    return result


def devices(selection):
    import subprocess

    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    physical = dict(line.strip().split(", ") for line in rows.splitlines() if line.strip())

    def resolve(tokens):
        result = []
        for token in tokens.split(","):
            matches = [uuid for index, uuid in physical.items() if token.strip() in {index, uuid}]
            if len(matches) != 1:
                raise ValueError(f"Unknown physical GPU: {token!r}")
            result.append(matches[0])
        if len(set(result)) != len(result):
            raise ValueError("Duplicate GPU selection")
        return result

    visible = (
        resolve(os.environ["CUDA_VISIBLE_DEVICES"])
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else list(physical.values())
    )
    selected = resolve(selection) if selection is not None else visible
    if not selected or not set(selected) <= set(visible):
        raise ValueError("Select GPUs within CUDA_VISIBLE_DEVICES")
    return selected


def owns_listener(pid, port):
    root = Path("/proc") / str(pid)
    try:
        inodes = {
            row.split()[9]
            for row in (root / "net/tcp").read_text().splitlines()[1:]
            if row.split()[3] == "0A" and int(row.split()[1].split(":")[1], 16) == port
        }
        for descriptor in (root / "fd").iterdir():
            try:
                if os.readlink(descriptor) in {f"socket:[{inode}]" for inode in inodes}:
                    return True
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    return False


def child_command(settings, parent, port):
    child = settings.model_copy(update={"port": port, "device": "cuda:0"})
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child-parent",
        str(parent),
        *serving_argv(child),
    ]


def run_child(args):
    if os.getppid() != args._child_parent:
        raise RuntimeError("Owning supervisor exited")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != args._child_parent:
        raise RuntimeError("Owning supervisor exited during bootstrap")
    import uvicorn

    from gs1.api.app import create_app

    settings = serving_settings(args)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


async def supervise(args, gpus):
    from gs1.model.util import checkpoint_files, file_hash

    settings = serving_settings(args)
    stop, replicas = asyncio.Event(), {}
    inputs = checkpoint_files(settings.artifact)
    inputs += [p for p in (settings.runtime_profile, settings.graph_warmup_requests) if p is not None]
    original = {str(p): file_hash(p) for p in inputs}
    state = dict(ready=False, stopped=False, replicas=[], input_sha256=original)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)

    def save():
        temporary = args.runs_dir / "supervisor.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.runs_dir / "supervisor.json")

    async def launch(index, generation):
        if original != {str(p): file_hash(p) for p in inputs}:
            raise ValueError("GS1 deployment inputs changed; refusing a different replacement")
        port = args.base_port + index
        log = (args.runs_dir / f"gpu{index}-generation{generation}.log").open("x")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpus[index],
            "OMP_NUM_THREADS": str(settings.cpu_threads),
            "MKL_NUM_THREADS": str(settings.cpu_threads),
        }
        try:
            process = await asyncio.create_subprocess_exec(
                *child_command(settings, os.getpid(), port),
                env=env,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        record = dict(
            gpu_uuid=gpus[index],
            port=port,
            pid=process.pid,
            generation=generation,
            ready=False,
            started_monotonic=time.monotonic(),
        )
        replicas[index] = process, log, record
        state["replicas"].append(record)
        save()

    async def drain(item):
        process, log, record = item
        try:
            for signum, timeout in (
                (signal.SIGINT, args.shutdown_timeout),
                (signal.SIGTERM, 10),
                (signal.SIGKILL, 5),
            ):
                if process.returncode is not None:
                    break
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout)
                except TimeoutError:
                    record["forced_termination"] = True
            await process.wait()
        finally:
            record.update(ready=False, exit_code=process.returncode)
            log.close()

    try:
        for index in range(len(gpus)):
            if stop.is_set():
                break
            await launch(index, 0)
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            while not stop.is_set():
                for index, (process, log, record) in list(replicas.items()):
                    if process.returncode is not None:
                        record.update(ready=False, exit_code=process.returncode)
                        state["ready"] = False
                        log.close()
                        if record["generation"] >= args.max_restarts:
                            raise RuntimeError(f"GPU {gpus[index]} restart budget exhausted")
                        try:
                            await asyncio.wait_for(stop.wait(), min(30, 2 ** record["generation"]))
                        except TimeoutError:
                            pass
                        if not stop.is_set():
                            await launch(index, record["generation"] + 1)
                        continue
                    ready = False
                    try:
                        response = await client.get(f"http://127.0.0.1:{record['port']}/v1/models")
                        ready = (
                            response.status_code == 200
                            and bool(response.json().get("models"))
                            and owns_listener(process.pid, record["port"])
                        )
                    except (httpx.HTTPError, ValueError):
                        pass
                    if ready:
                        record.pop("unhealthy_since", None)
                        record["ever_ready"] = True
                    else:
                        since = record.setdefault(
                            "unhealthy_since",
                            time.monotonic() if record.get("ever_ready") else record["started_monotonic"],
                        )
                        timeout = args.unhealthy_timeout if record.get("ever_ready") else args.startup_timeout
                        if time.monotonic() - since >= timeout:
                            raise RuntimeError(f"GPU {gpus[index]} readiness timeout")
                    record["ready"] = ready
                was_ready = state["ready"]
                state["ready"] = len(replicas) == len(gpus) and all(
                    r["ready"] for _, _, r in replicas.values()
                )
                save()
                if state["ready"] and not was_ready:
                    print(
                        json.dumps(
                            dict(
                                ready=True,
                                endpoints=[f"http://127.0.0.1:{r['port']}" for _, _, r in replicas.values()],
                            )
                        ),
                        flush=True,
                    )
                try:
                    await asyncio.wait_for(stop.wait(), 1)
                except TimeoutError:
                    pass
    finally:
        state["ready"] = False
        results = await asyncio.gather(*(drain(item) for item in replicas.values()), return_exceptions=True)
        state["stopped"] = True
        state["clean_shutdown"] = all(not isinstance(r, BaseException) for r in results) and all(
            not r.get("forced_termination") and r.get("exit_code") == 0 for _, _, r in replicas.values()
        )
        try:
            save()
        finally:
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(signum)
        if not state["clean_shutdown"]:
            raise RuntimeError("Unclean GS1 replica shutdown; inspect supervisor.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_serving_arguments(parser)
    parser.set_defaults(host="127.0.0.1")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/replicas"))
    parser.add_argument("--gpus")
    parser.add_argument("--base-port", type=int, default=18080)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--unhealthy-timeout", type=float, default=30)
    parser.add_argument("--shutdown-timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_child-parent", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    import math

    if not 0 <= args.max_restarts <= 10 or not all(
        math.isfinite(v) and v > 0
        for v in (args.startup_timeout, args.unhealthy_timeout, args.shutdown_timeout)
    ):
        parser.error("Require finite positive timeouts and a bounded restart count")
    if args.host != "127.0.0.1":
        parser.error("The replica supervisor binds localhost")
    settings = serving_settings(args)
    from gs1.model.util import resolve_source

    args.artifact = resolve_source(settings.artifact)
    args.runs_dir = args.runs_dir.resolve()
    for name in ("runtime_profile", "graph_warmup_requests"):
        if getattr(settings, name) is not None:
            setattr(args, name, getattr(settings, name).resolve())
    if args._child_parent is not None:
        run_child(args)
        return
    if args.device not in {"cuda", "cuda:0"}:
        parser.error("Each supervised replica uses one GPU as cuda:0")
    gpus = devices(args.gpus)
    if not 1 <= args.base_port <= 65536 - len(gpus):
        parser.error("GPU count exceeds the valid port range")
    settings = serving_settings(args)
    if args.dry_run:
        print(
            json.dumps(
                [
                    dict(
                        gpu=gpu,
                        command=child_command(
                            settings,
                            os.getpid(),
                            args.base_port + i,
                        ),
                    )
                    for i, gpu in enumerate(gpus)
                ],
                indent=2,
            )
        )
        return
    args.runs_dir.mkdir(parents=True, exist_ok=False)
    with ExitStack() as locks:
        directory = Path(tempfile.gettempdir()) / f"gs1-device-locks-{os.getuid()}"
        directory.mkdir(exist_ok=True)
        for uuid in sorted(gpus):
            handle = locks.enter_context((directory / uuid).open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(supervise(args, gpus))


if __name__ == "__main__":
    main()
