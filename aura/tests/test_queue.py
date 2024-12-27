import pytest
import time
from aura.queue import QueueManager, Priority, UserState, QueueError

@pytest.fixture
def queue_manager():
    return QueueManager(5, 100, 3)

def test_queue_initialization(queue_manager):
    assert queue_manager.max_session_time == 300
    assert queue_manager.max_queue_size == 100
    assert queue_manager.max_reconnect_attempts == 3

def test_add_to_queue(queue_manager):
    position = queue_manager.add_to_queue("user1", Priority.normal())
    assert position == 1

def test_priority_ordering(queue_manager):
    queue_manager.add_to_queue("user1", Priority.normal())
    queue_manager.add_to_queue("user2", Priority.high())
    queue_manager.add_to_queue("user3", Priority.low())
    
    queue_list = list(queue_manager.queue)
    assert queue_list[0].id == "user2"  # High priority first
    assert queue_list[1].id == "user1"  # Normal priority second
    assert queue_list[2].id == "user3"  # Low priority last

def test_duplicate_user(queue_manager):
    queue_manager.add_to_queue("user1", None)
    with pytest.raises(QueueError) as exc_info:
        queue_manager.add_to_queue("user1", None)
    assert "already in queue" in str(exc_info.value).lower()

def test_queue_full():
    queue = QueueManager(5, 2, 3)
    queue.add_to_queue("user1", None)
    queue.add_to_queue("user2", None)
    with pytest.raises(QueueError) as exc_info:
        queue.add_to_queue("user3", None)
    assert "queue is full" in str(exc_info.value).lower()

def test_state_transitions(queue_manager):
    queue_manager.add_to_queue("user1", None)
    queue_manager.update_user_state("user1", UserState.CONNECTING)
    queue_manager.update_user_state("user1", UserState.CONNECTED)
    queue_manager.update_user_state("user1", UserState.DISCONNECTED)

def test_cleanup_timeouts(queue_manager):
    queue_manager.add_to_queue("user1", None)
    # Force timeout by waiting
    time.sleep(31)  
    timed_out = queue_manager.cleanup_timeouts()
    assert len(timed_out) == 1
    assert timed_out[0] == "user1"

def test_remove_from_queue(queue_manager):
    queue_manager.add_to_queue("user1", None)
    assert queue_manager.remove_from_queue("user1") is True
    assert queue_manager.remove_from_queue("nonexistent") is False

def test_invalid_state_transition(queue_manager):
    queue_manager.add_to_queue("user1", None)
    with pytest.raises(QueueError) as exc_info:
        queue_manager.update_user_state("user1", UserState.DISCONNECTED)
    assert "invalid state transition" in str(exc_info.value).lower()

def test_user_not_found(queue_manager):
    with pytest.raises(QueueError) as exc_info:
        queue_manager.update_user_state("nonexistent", UserState.CONNECTED)
    assert "not found" in str(exc_info.value).lower()
