"""Priority session queue for Aura WebRTC viewers."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional


class QueueError(Exception):
    """Raised for invalid queue operations."""


class UserState(Enum):
    WAITING = "WAITING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class Priority(Enum):
    HIGH = 3
    NORMAL = 2
    LOW = 1

    @classmethod
    def high(cls) -> "Priority":
        return cls.HIGH

    @classmethod
    def normal(cls) -> "Priority":
        return cls.NORMAL

    @classmethod
    def low(cls) -> "Priority":
        return cls.LOW


@dataclass
class QueuedUser:
    id: str
    state: UserState = UserState.WAITING
    join_time: float = field(default_factory=time.time)
    priority: Priority = Priority.NORMAL
    last_activity: float = field(default_factory=time.time)
    connection_attempts: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    def __init__(
        self, user_id: str, priority: Optional[Priority] = None
    ) -> None:
        self.id = user_id
        self.state = UserState.WAITING
        self.join_time = time.time()
        self.priority = priority or Priority.NORMAL
        self.last_activity = time.time()
        self.connection_attempts = 0
        self.metadata = {}


class QueueStats:
    def __init__(self) -> None:
        self.total_users = 0
        self.waiting_users = 0
        self.connected_users = 0
        self.average_wait_time = 0.0
        self.max_wait_time = 0.0


class QueueManager:
    """In-memory priority queue for interactive Aura sessions."""

    def __init__(
        self,
        max_session_minutes: int = 5,
        max_queue_size: int = 100,
        max_reconnect_attempts: int = 3,
    ) -> None:
        self.queue: Deque[QueuedUser] = deque()
        self.current_user: Optional[QueuedUser] = None
        self.max_session_time = max_session_minutes * 60
        self.max_queue_size = max_queue_size
        self.max_reconnect_attempts = max_reconnect_attempts
        self.user_timeouts: Dict[str, float] = {}
        self.stats = QueueStats()

    def add_to_queue(
        self, user_id: str, priority: Optional[Priority] = None
    ) -> int:
        if len(self.queue) >= self.max_queue_size:
            raise QueueError("Queue is full")
        if any(user.id == user_id for user in self.queue):
            raise QueueError(f"User {user_id} already in queue")

        user = QueuedUser(user_id, priority)
        self._update_stats(user)

        insert_pos = len(self.queue)
        for index, existing in enumerate(self.queue):
            if existing.priority.value <= user.priority.value:
                insert_pos = index
                break

        self.queue.insert(insert_pos, user)
        return insert_pos + 1

    def remove_from_queue(self, user_id: str) -> bool:
        for user in list(self.queue):
            if user.id == user_id:
                self.queue.remove(user)
                return True
        return False

    def update_user_state(self, user_id: str, new_state: UserState) -> None:
        for user in self.queue:
            if user.id != user_id:
                continue
            if not self._is_valid_state_transition(user.state, new_state):
                raise QueueError(
                    f"Invalid state transition from {user.state} to {new_state}"
                )
            user.state = new_state
            user.last_activity = time.time()
            return
        raise QueueError(f"User {user_id} not found")

    def cleanup_timeouts(self, timeout_duration: float = 30.0) -> List[str]:
        current_time = time.time()
        timed_out: List[str] = []
        remaining: Deque[QueuedUser] = deque()
        for user in self.queue:
            if current_time - user.last_activity > timeout_duration:
                timed_out.append(user.id)
            else:
                remaining.append(user)
        self.queue = remaining
        return timed_out

    def get_queue_stats(self) -> str:
        return json.dumps(
            {
                "total_users": self.stats.total_users,
                "waiting_users": self.stats.waiting_users,
                "connected_users": self.stats.connected_users,
                "average_wait_time": self.stats.average_wait_time,
                "max_wait_time": self.stats.max_wait_time,
            }
        )

    def _update_stats(self, user: QueuedUser) -> None:
        self.stats.total_users += 1
        self.stats.waiting_users = len(self.queue)
        wait_time = time.time() - user.join_time
        self.stats.average_wait_time = (
            self.stats.average_wait_time + wait_time
        ) / 2
        self.stats.max_wait_time = max(self.stats.max_wait_time, wait_time)

    def _is_valid_state_transition(
        self, from_state: UserState, to_state: UserState
    ) -> bool:
        valid = {
            UserState.WAITING: [UserState.CONNECTING],
            UserState.CONNECTING: [UserState.CONNECTED, UserState.ERROR],
            UserState.CONNECTED: [UserState.DISCONNECTED, UserState.TIMEOUT],
            UserState.DISCONNECTED: [],
            UserState.TIMEOUT: [],
            UserState.ERROR: [],
        }
        return to_state in valid.get(from_state, [])
