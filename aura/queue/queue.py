from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, List, Optional, Deque
from collections import deque
import json

class QueueError(Exception):
    pass

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
    def high(cls):
        return cls.HIGH

    @classmethod
    def normal(cls):
        return cls.NORMAL

    @classmethod
    def low(cls):
        return cls.LOW

@dataclass
class QueuedUser:
    id: str
    state: UserState
    join_time: float
    priority: Priority
    last_activity: float
    connection_attempts: int
    metadata: Dict[str, str]

    def __init__(self, id: str, priority: Optional[Priority] = None):
        self.id = id
        self.state = UserState.WAITING
        self.join_time = time.time()
        self.priority = priority or Priority.NORMAL
        self.last_activity = time.time()
        self.connection_attempts = 0
        self.metadata = {}

class QueueStats:
    def __init__(self):
        self.total_users = 0
        self.waiting_users = 0
        self.connected_users = 0
        self.average_wait_time = 0
        self.max_wait_time = 0

class QueueManager:
    def __init__(self, max_session_minutes: int = 5, max_queue_size: int = 100, max_reconnect_attempts: int = 3):
        self.queue: Deque[QueuedUser] = deque()
        self.current_user: Optional[QueuedUser] = None
        self.max_session_time = max_session_minutes * 60
        self.max_queue_size = max_queue_size
        self.max_reconnect_attempts = max_reconnect_attempts
        self.user_timeouts: Dict[str, float] = {}
        self.stats = QueueStats()

    def add_to_queue(self, user_id: str, priority: Optional[Priority] = None) -> int:
        if len(self.queue) >= self.max_queue_size:
            raise QueueError("Queue is full")
        
        if any(u.id == user_id for u in self.queue):
            raise QueueError(f"User {user_id} already in queue")

        user = QueuedUser(user_id, priority)
        self._update_stats(user)

        insert_pos = len(self.queue)
        for i, u in enumerate(self.queue):
            if u.priority.value <= user.priority.value:
                insert_pos = i
                break

        self.queue.insert(insert_pos, user)
        return insert_pos + 1

    def remove_from_queue(self, user_id: str) -> bool:
        for i, user in enumerate(self.queue):
            if user.id == user_id:
                self.queue.remove(user)
                return True
        return False

    def update_user_state(self, user_id: str, new_state: UserState):
        for user in self.queue:
            if user.id == user_id:
                if not self._is_valid_state_transition(user.state, new_state):
                    raise QueueError(f"Invalid state transition from {user.state} to {new_state}")
                user.state = new_state
                user.last_activity = time.time()
                return
        raise QueueError(f"User {user_id} not found")

    def cleanup_timeouts(self) -> List[str]:
        timeout_duration = 30 
        current_time = time.time()
        timed_out = []
        
        self.queue = deque([user for user in self.queue if not (
            current_time - user.last_activity > timeout_duration and 
            not timed_out.append(user.id)
        )])
        
        return timed_out

    def get_queue_stats(self) -> str:
        return json.dumps({
            "total_users": self.stats.total_users,
            "waiting_users": self.stats.waiting_users,
            "connected_users": self.stats.connected_users,
            "average_wait_time": self.stats.average_wait_time,
            "max_wait_time": self.stats.max_wait_time
        })

    def _update_stats(self, user: QueuedUser):
        self.stats.total_users += 1
        self.stats.waiting_users = len(self.queue)
        wait_time = time.time() - user.join_time
        self.stats.average_wait_time = (self.stats.average_wait_time + wait_time) / 2
        if wait_time > self.stats.max_wait_time:
            self.stats.max_wait_time = wait_time

    def _is_valid_state_transition(self, from_state: UserState, to_state: UserState) -> bool:
        valid_transitions = {
            UserState.WAITING: [UserState.CONNECTING],
            UserState.CONNECTING: [UserState.CONNECTED, UserState.ERROR],
            UserState.CONNECTED: [UserState.DISCONNECTED, UserState.TIMEOUT],
            UserState.DISCONNECTED: [],
            UserState.TIMEOUT: [],
            UserState.ERROR: []
        }
        return to_state in valid_transitions.get(from_state, [])
