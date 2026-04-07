from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class AgentRun:
    run_id: str
    session_id: str
    status: str
    accepted_at: float
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None


class _SessionLane:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.queue_depth = 0


class AgentRuntime:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._runs: dict[str, AgentRun] = {}
        self._run_events: dict[str, threading.Event] = {}
        self._lanes: dict[str, _SessionLane] = {}

    def create_run(self, session_id: str) -> AgentRun:
        run = AgentRun(
            run_id=str(uuid.uuid4()),
            session_id=session_id,
            status="accepted",
            accepted_at=time.time(),
        )
        with self._guard:
            self._runs[run.run_id] = run
            self._run_events[run.run_id] = threading.Event()
            self._lanes.setdefault(session_id, _SessionLane())
        return run

    @contextmanager
    def session_lane(self, session_id: str, run_id: str):
        lane = self._get_lane(session_id)
        with self._guard:
            lane.queue_depth += 1
        lane.lock.acquire()
        try:
            with self._guard:
                lane.queue_depth = max(0, lane.queue_depth - 1)
                run = self._runs.get(run_id)
                if run is not None:
                    run.status = "running"
                    run.started_at = time.time()
            yield
        finally:
            lane.lock.release()

    def finish_run(self, run_id: str, *, status: str, error: str | None = None) -> None:
        with self._guard:
            run = self._runs.get(run_id)
            event = self._run_events.get(run_id)
            if run is None:
                return
            run.status = status
            run.ended_at = time.time()
            run.error = error
            if event is not None:
                event.set()

    def get_run(self, run_id: str) -> AgentRun | None:
        with self._guard:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return AgentRun(**run.__dict__)

    def wait_for_run(self, run_id: str, timeout_ms: int = 30_000) -> dict[str, object]:
        with self._guard:
            run = self._runs.get(run_id)
            event = self._run_events.get(run_id)
            if run is None or event is None:
                return {"status": "missing", "runId": run_id}
        completed = event.wait(timeout_ms / 1000)
        snapshot = self.get_run(run_id)
        if snapshot is None:
            return {"status": "missing", "runId": run_id}
        if not completed:
            return {
                "status": "timeout",
                "runId": snapshot.run_id,
                "startedAt": snapshot.started_at,
                "endedAt": snapshot.ended_at,
            }
        return {
            "status": snapshot.status,
            "runId": snapshot.run_id,
            "startedAt": snapshot.started_at,
            "endedAt": snapshot.ended_at,
            "error": snapshot.error,
        }

    def lane_snapshot(self, session_id: str) -> dict[str, int]:
        lane = self._get_lane(session_id)
        with self._guard:
            queue_depth = lane.queue_depth
            active = 1 if lane.lock.locked() else 0
        return {"active": active, "queued": queue_depth}

    def _get_lane(self, session_id: str) -> _SessionLane:
        with self._guard:
            return self._lanes.setdefault(session_id, _SessionLane())
