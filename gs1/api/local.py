"""Bounded GS1 slots with thread-owned GPU resources and cancellation-safe draining."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass


class BackendError(Exception):
    def __init__(self, message, status=529):
        super().__init__(message)
        self.status = status


@dataclass
class PendingRequest:
    request: object
    tenant: str
    deadline: float
    future: object
    cancelled: object


class LocalBackend:
    def __init__(self, settings):
        self.settings = settings
        self.engines, self.streams = [], []
        self.metadata = {}
        self.reason = "GS1 model is not loaded"
        self.slots = settings.graph_slots if settings.cuda_graphs else 1
        self.executors = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gs1-gpu-{i}") for i in range(self.slots)
        ]
        self.queue = asyncio.Queue(maxsize=settings.max_queue)
        self.workers = []
        self._inflight = set()
        self._failure = None
        self._closing = False
        self._close_task = None

    @property
    def available(self):
        return bool(self.engines) and not self._closing and self._failure is None

    def _load(self):
        import torch

        from gs1.inference.engine import Engine

        s = self.settings
        torch.set_num_threads(s.cpu_threads)
        self.loaded = Engine.from_pretrained(
            s.artifact,
            device=s.device,
            precision=s.precision,
            mode=s.engine,
            runtime_profile=s.runtime_profile,
        )
        self.metadata = self.loaded.identity
        self.profile = self.loaded.runtime_profile
        self.qualified = self.profile is not None
        if s.cuda_graphs and (
            self.profile is None or not self.profile.data["engines"][s.engine].get("graphs_passed")
        ):
            raise ValueError("The GS1 runtime profile does not qualify CUDA graphs")
        self.policy = self.loaded.execution_policy

    def _make_slot(self, index):
        import json

        import torch

        from gs1.inference.engine import Engine
        from gs1.model.schema import DecisionRequest

        s, metadata = self.settings, self.metadata
        compiler = self.loaded.model.decision_compiler(
            max_context=s.max_context,
            max_questions=s.max_questions,
            max_request_tokens=s.max_request_tokens,
            max_branches=s.max_branches,
        )
        graph = None
        policy = self.policy
        if s.cuda_graphs:
            from gs1.inference.cuda_graph import NativeGraphCache

            graph = NativeGraphCache(
                self.loaded.model,
                s.precision,
                max_bytes=s.graph_cache_mb * 1024**2 // self.slots,
                max_entries=s.graph_cache_entries // self.slots,
                execution_policy=policy,
            )
        engine = Engine(
            self.loaded.model,
            compiler,
            s.engine,
            min(s.batch_tokens, metadata["limits"]["batch_tokens"]),
            s.cache_mb / self.slots,
            identity=metadata,
            precision=s.precision,
            reference_batching=self.qualified,
            graph_cache=graph,
            execution_policy=policy,
            runtime_profile=self.profile,
            max_attention_bytes=min(s.max_attention_bytes, metadata["limits"]["max_attention_bytes"]),
            max_logit_elements=min(s.max_logit_elements, metadata["limits"]["max_logit_elements"]),
        )
        stream = torch.cuda.Stream(device=s.device) if torch.device(s.device).type == "cuda" else None
        try:
            if graph is not None and s.graph_capture_policy == "prewarm":
                if s.graph_warmup_requests is None:
                    raise ValueError("Graph prewarm requires --graph-warmup-requests")
                values = json.loads(s.graph_warmup_requests.read_text())
                if not isinstance(values, list) or not values:
                    raise ValueError("Graph warmup must contain a nonempty list of GS1 requests")
                with torch.cuda.stream(stream):
                    for value in values:
                        engine.evaluate(DecisionRequest.model_validate(value), metadata["resolved_model"])
                if graph.stats["rejects"] or not graph.entries:
                    raise ValueError("GS1 prewarm did not pass exact capture and residency checks")
                graph.allow_capture = False
                before = graph.stats["misses"]
                with torch.cuda.stream(stream):
                    for value in values:
                        engine.cache.entries.clear()
                        engine.cache.bytes = 0
                        engine.evaluate(DecisionRequest.model_validate(value), metadata["resolved_model"])
                if graph.stats["misses"] != before:
                    raise ValueError("GS1 graph prewarm exceeds its resident entry/byte budget")
            return engine, stream
        except BaseException:
            engine.close()
            raise

    async def start(self):
        if self.workers or self._closing:
            raise RuntimeError("GS1 backend cannot be restarted")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executors[0], self._load)
            for index, executor in enumerate(self.executors):
                engine, stream = await loop.run_in_executor(executor, self._make_slot, index)
                self.engines.append(engine)
                self.streams.append(stream)
            for index in range(self.slots):
                worker = asyncio.create_task(self._work(index), name=f"gs1-slot-{index}")
                worker.add_done_callback(self._worker_finished)
                self.workers.append(worker)
            self.reason = ""
        except BaseException as error:
            self.reason = f"Checkpoint unavailable: {error}"
            await self.close()
            raise

    def _fail(self, error):
        if self._failure is not None or self._closing:
            return
        self._failure = str(error)[:1024]
        self.reason = "GS1 execution worker failed: " + self._failure
        self._drain(self.reason)
        current = asyncio.current_task()
        for worker in self.workers:
            if worker is not current and not worker.done():
                worker.cancel()

    def _worker_finished(self, worker):
        if self._closing or self._failure:
            return
        error = (
            BackendError("Execution worker was cancelled unexpectedly")
            if worker.cancelled()
            else (worker.exception() or BackendError("Execution worker exited unexpectedly"))
        )
        self._fail(error)

    def _drain(self, message):
        for future in tuple(self._inflight):
            if not future.done():
                future.set_exception(BackendError(message))
        while not self.queue.empty():
            pending = self.queue.get_nowait()
            pending.cancelled.set()
            self.queue.task_done()

    def _execute(self, index, pending):
        import torch

        engine, stream = self.engines[index], self.streams[index]
        result, failure, fatal = None, None, False
        try:
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                result = engine.evaluate(
                    pending.request,
                    self.metadata["resolved_model"],
                    tenant=pending.tenant,
                    cancelled=lambda: pending.cancelled.is_set() or time.monotonic() >= pending.deadline,
                )
                if stream is not None:
                    stream.synchronize()
        except ValueError as error:
            failure = (str(error)[:1024], 422)
        except TimeoutError as error:
            failure = (str(error)[:1024], 504)
        except FloatingPointError as error:
            failure = (str(error)[:1024], 529)
        except BaseException as error:
            failure, fatal = (f"{type(error).__name__}: {error}"[:1024], 529), True
        # A future must never retain a GPU traceback or a request's tensor graph.
        return result, failure, fatal

    async def _work(self, index):
        while True:
            pending = await self.queue.get()
            execution = None
            try:
                if pending.future.done():
                    continue
                if time.monotonic() >= pending.deadline:
                    pending.future.set_exception(BackendError("Request expired in the queue", 504))
                    continue
                execution = asyncio.get_running_loop().run_in_executor(
                    self.executors[index], self._execute, index, pending
                )
                response, failure, fatal = await asyncio.shield(execution)
                if fatal:
                    self._fail(BackendError(*failure))
                    return
                if not pending.future.done():
                    if failure is not None:
                        pending.future.set_exception(BackendError(*failure))
                    else:
                        pending.future.set_result(response)
            except asyncio.CancelledError:
                pending.cancelled.set()
                if execution is not None:
                    await asyncio.shield(execution)
                raise
            finally:
                self.queue.task_done()
                pending = execution = None

    async def evaluate(self, request, tenant="local"):
        if not self.available:
            raise BackendError(self.reason)
        if self.queue.full():
            raise BackendError("Local inference queue is full", 529)
        future = asyncio.get_running_loop().create_future()
        token = threading.Event()
        queued_at = time.monotonic()
        pending = PendingRequest(request, tenant, queued_at + self.settings.deadline_seconds, future, token)
        future.add_done_callback(lambda f: token.set() if f.cancelled() else None)
        self._inflight.add(future)
        future.add_done_callback(self._inflight.discard)
        self.queue.put_nowait(pending)
        try:
            return await asyncio.wait_for(future, self.settings.deadline_seconds)
        except TimeoutError:
            raise BackendError("Local inference deadline exceeded", 504) from None

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await self._close_task
            raise

    async def _finish_close(self):
        self._closing = True
        self._drain("Server is shutting down")
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        try:
            for index, engine in enumerate(self.engines):
                await asyncio.get_running_loop().run_in_executor(self.executors[index], engine.close)
        finally:
            for executor in self.executors:
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)
                self.engines.clear()
            self.streams.clear()
            self.loaded = self.policy = None
