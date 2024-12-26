use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::Mutex;

#[derive(Debug)]
pub enum QueueError {
    AlreadyInQueue(String),
    UserNotFound(String),
    QueueFull,
    InvalidStateTransition(UserState, UserState),
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueueError::AlreadyInQueue(user) => write!(f, "User {} already in queue", user),
            QueueError::UserNotFound(user) => write!(f, "User {} not found", user),
            QueueError::QueueFull => write!(f, "Queue is full"),
            QueueError::InvalidStateTransition(from, to) => {
                write!(f, "Invalid state transition from {:?} to {:?}", from, to)
            }
        }
    }
}

impl std::error::Error for QueueError {}

impl From<QueueError> for PyErr {
    fn from(err: QueueError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UserState {
    WAITING(),
    CONNECTING(),
    CONNECTED(),
    DISCONNECTED(),
    TIMEOUT(),
    ERROR(String),
}

#[pymethods]
impl UserState {
    #[staticmethod]
    fn from_str(state: &str) -> PyResult<Self> {
        match state {
            "WAITING" => Ok(UserState::WAITING()),
            "CONNECTING" => Ok(UserState::CONNECTING()),
            "CONNECTED" => Ok(UserState::CONNECTED()),
            "DISCONNECTED" => Ok(UserState::DISCONNECTED()),
            "TIMEOUT" => Ok(UserState::TIMEOUT()),
            s if s.starts_with("ERROR: ") => Ok(UserState::ERROR(s[7..].to_string())),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid state",
            )),
        }
    }

    fn __str__(&self) -> PyResult<String> {
        match self {
            UserState::WAITING() => Ok("WAITING".to_string()),
            UserState::CONNECTING() => Ok("CONNECTING".to_string()),
            UserState::CONNECTED() => Ok("CONNECTED".to_string()),
            UserState::DISCONNECTED() => Ok("DISCONNECTED".to_string()),
            UserState::TIMEOUT() => Ok("TIMEOUT".to_string()),
            UserState::ERROR(msg) => Ok(format!("ERROR: {}", msg)),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct QueuedUser {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub state: UserState,
    #[pyo3(get)]
    pub join_time: SystemTime,
    #[pyo3(get)]
    pub priority: Priority,
    #[pyo3(get)]
    pub last_activity: SystemTime,
    #[pyo3(get)]
    pub connection_attempts: u32,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl QueuedUser {
    #[new]
    #[pyo3(signature = (id, priority=None))]
    fn new(id: String, priority: Option<Priority>) -> Self {
        QueuedUser {
            id,
            state: UserState::WAITING(),
            join_time: SystemTime::now(),
            priority: priority.unwrap_or(Priority::Normal),
            last_activity: SystemTime::now(),
            connection_attempts: 0,
            metadata: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[pyclass(eq, eq_int)]
pub enum Priority {
    #[pyo3(name = "HIGH")]
    High,
    #[pyo3(name = "NORMAL")]
    Normal,
    #[pyo3(name = "LOW")]
    Low,
}

#[pymethods]
impl Priority {
    #[staticmethod]
    fn high() -> Self {
        Priority::High
    }

    #[staticmethod]
    fn normal() -> Self {
        Priority::Normal
    }

    #[staticmethod]
    fn low() -> Self {
        Priority::Low
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    total_users: usize,
    waiting_users: usize,
    connected_users: usize,
    average_wait_time: Duration,
    max_wait_time: Duration,
}

#[pyclass]
pub struct QueueManager {
    pub queue: Arc<Mutex<VecDeque<QueuedUser>>>,
    pub current_user: Arc<Mutex<Option<QueuedUser>>>,
    pub max_session_time: Duration,
    pub max_queue_size: usize,
    pub max_reconnect_attempts: u32,
    pub user_timeouts: HashMap<String, SystemTime>,
    pub stats: Arc<Mutex<QueueStats>>,
}

#[pymethods]
impl QueueManager {
    #[new]
    #[pyo3(signature = (max_session_minutes=5, max_queue_size=100, max_reconnect_attempts=3))]
    pub fn new(
        max_session_minutes: u64,
        max_queue_size: usize,
        max_reconnect_attempts: u32,
    ) -> Self {
        QueueManager {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            current_user: Arc::new(Mutex::new(None)),
            max_session_time: Duration::from_secs(max_session_minutes * 60),
            max_queue_size,
            max_reconnect_attempts,
            user_timeouts: HashMap::new(),
            stats: Arc::new(Mutex::new(QueueStats {
                total_users: 0,
                waiting_users: 0,
                connected_users: 0,
                average_wait_time: Duration::default(),
                max_wait_time: Duration::default(),
            })),
        }
    }

    #[pyo3(signature = (user_id, priority=None))]
    #[pyo3(text_signature = "(self, user_id: str, priority: Optional[Priority] = None) -> int")]
    pub fn add_to_queue(&self, user_id: String, priority: Option<Priority>) -> PyResult<usize> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut queue = self.queue.lock().await;

            if queue.len() >= self.max_queue_size {
                return Err(QueueError::QueueFull.into());
            }

            if queue.iter().any(|u| u.id == user_id) {
                return Err(QueueError::AlreadyInQueue(user_id).into());
            }

            let position = queue.len();
            let user = QueuedUser {
                id: user_id,
                state: UserState::WAITING(),
                join_time: SystemTime::now(),
                priority: priority.unwrap_or(Priority::Normal),
                last_activity: SystemTime::now(),
                connection_attempts: 0,
                metadata: HashMap::new(),
            };

            self.update_stats(&user.clone())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let insert_pos = queue
                .iter()
                .position(|u| u.priority <= user.priority)
                .unwrap_or(queue.len());
            queue.insert(insert_pos, user);

            Ok(position + 1)
        })
    }

    pub fn remove_from_queue(&self, user_id: &str) -> PyResult<bool> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut queue = self.queue.lock().await;
            if let Some(pos) = queue.iter().position(|u| u.id == user_id) {
                queue.remove(pos);
                Ok(true)
            } else {
                Ok(false)
            }
        })
    }

    pub fn update_user_state(&self, user_id: &str, new_state: UserState) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut queue = self.queue.lock().await;
            let mut current = self.current_user.lock().await;

            if let Some(user) = queue.iter_mut().find(|u| u.id == user_id) {
                if !self.is_valid_state_transition(&user.state, &new_state) {
                    return Err(
                        QueueError::InvalidStateTransition(user.state.clone(), new_state).into(),
                    );
                }
                user.state = new_state;
                user.last_activity = SystemTime::now();
                Ok(())
            } else if let Some(ref mut current_user) = *current {
                if current_user.id == user_id {
                    current_user.state = new_state;
                    current_user.last_activity = SystemTime::now();
                    Ok(())
                } else {
                    Err(QueueError::UserNotFound(user_id.to_string()).into())
                }
            } else {
                Err(QueueError::UserNotFound(user_id.to_string()).into())
            }
        })
    }

    pub fn get_queue_stats(&self) -> PyResult<String> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let stats = self.stats.lock().await;
            match serde_json::to_string(&*stats) {
                Ok(json) => Ok(json),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e.to_string(),
                )),
            }
        })
    }

    pub fn cleanup_timeouts(&self) -> PyResult<Vec<String>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut queue = self.queue.lock().await;
            let now = SystemTime::now();
            let timeout_duration = Duration::from_secs(30);

            let timed_out: Vec<String> = queue
                .iter()
                .filter(|u| u.last_activity.elapsed().unwrap_or_default() > timeout_duration)
                .map(|u| u.id.clone())
                .collect();

            queue.retain(|u| !timed_out.contains(&u.id));
            Ok(timed_out)
        })
    }

    fn update_stats(&self, user: &QueuedUser) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        rt.block_on(async {
            let mut stats = self.stats.lock().await;
            stats.total_users += 1;
            stats.waiting_users = self.queue.lock().await.len();

            if let Ok(wait_time) = user.join_time.elapsed() {
                stats.average_wait_time = (stats.average_wait_time + wait_time) / 2;
                if wait_time > stats.max_wait_time {
                    stats.max_wait_time = wait_time;
                }
            }
            Ok(())
        })
    }

    fn is_valid_state_transition(&self, from: &UserState, to: &UserState) -> bool {
        matches!(
            (from, to),
            (UserState::WAITING(), UserState::CONNECTING())
                | (UserState::CONNECTING(), UserState::CONNECTED())
                | (UserState::CONNECTED(), UserState::DISCONNECTED())
                | (UserState::CONNECTING(), UserState::ERROR(_))
                | (UserState::CONNECTED(), UserState::TIMEOUT())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::runtime::Runtime;

    fn setup() -> QueueManager {
        QueueManager::new(5, 100, 3)
    }

    #[test]
    fn test_queue_initialization() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            assert_eq!(queue.queue.lock().await.len(), 0);
            assert!(queue.current_user.lock().await.is_none());
            assert_eq!(queue.max_session_time, Duration::from_secs(300));
            assert_eq!(queue.max_queue_size, 100);
            assert_eq!(queue.max_reconnect_attempts, 3);
        });
    }

    #[test]
    fn test_add_to_queue() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            let result = queue.add_to_queue("user1".to_string(), Some(Priority::Normal));
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), 1);

            let queue_length = queue.queue.lock().await.len();
            assert_eq!(queue_length, 1);
        });
    }

    #[test]
    fn test_priority_ordering() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue
                .add_to_queue("user1".to_string(), Some(Priority::Normal))
                .unwrap();
            queue
                .add_to_queue("user2".to_string(), Some(Priority::High))
                .unwrap();
            queue
                .add_to_queue("user3".to_string(), Some(Priority::Low))
                .unwrap();

            let users = queue.queue.lock().await;
            assert_eq!(users[0].id, "user2"); // High priority
            assert_eq!(users[1].id, "user1"); // Normal priority
            assert_eq!(users[2].id, "user3"); // Low priority
        });
    }

    #[test]
    fn test_duplicate_user() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue.add_to_queue("user1".to_string(), None).unwrap();
            let result = queue.add_to_queue("user1".to_string(), None);
            assert!(matches!(result.unwrap_err().to_string(), s if s.contains("already in queue")));
        });
    }

    #[test]
    fn test_queue_full() {
        let queue = QueueManager::new(5, 2, 3);
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue.add_to_queue("user1".to_string(), None).unwrap();
            queue.add_to_queue("user2".to_string(), None).unwrap();
            let result = queue.add_to_queue("user3".to_string(), None);
            assert!(matches!(result.unwrap_err().to_string(), s if s.contains("Queue is full")));
        });
    }

    #[test]
    fn test_state_transitions() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue.add_to_queue("user1".to_string(), None).unwrap();

            // Valid transitions
            assert!(queue
                .update_user_state("user1", UserState::CONNECTING())
                .is_ok());
            assert!(queue
                .update_user_state("user1", UserState::CONNECTED())
                .is_ok());
            assert!(queue
                .update_user_state("user1", UserState::DISCONNECTED())
                .is_ok());

            // Invalid transition
            let result = queue.update_user_state("user1", UserState::WAITING());
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_cleanup_timeouts() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue.add_to_queue("user1".to_string(), None).unwrap();

            std::thread::sleep(Duration::from_secs(31));

            let timed_out = queue.cleanup_timeouts().unwrap();
            assert_eq!(timed_out.len(), 1);
            assert_eq!(timed_out[0], "user1");

            let queue_length = queue.queue.lock().await.len();
            assert_eq!(queue_length, 0);
        });
    }

    #[test]
    fn test_queue_stats() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue.add_to_queue("user1".to_string(), None).unwrap();
            queue.add_to_queue("user2".to_string(), None).unwrap();

            let stats = queue.get_queue_stats().unwrap();
            assert!(stats.contains("\"total_users\":2"));
            assert!(stats.contains("\"waiting_users\":2"));
        });
    }

    #[test]
    fn test_remove_from_queue() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue.add_to_queue("user1".to_string(), None).unwrap();

            assert!(queue.remove_from_queue("user1").unwrap());
            assert!(!queue.remove_from_queue("nonexistent").unwrap());

            let queue_length = queue.queue.lock().await.len();
            assert_eq!(queue_length, 0);
        });
    }

    #[test]
    fn test_user_metadata() {
        let queue = setup();
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            queue.add_to_queue("user1".to_string(), None).unwrap();

            let mut queue_lock = queue.queue.lock().await;
            if let Some(user) = queue_lock.iter_mut().next() {
                user.metadata
                    .insert("browser".to_string(), "Chrome".to_string());
                assert_eq!(user.metadata.get("browser").unwrap(), "Chrome");
            }
        });
    }
}
