use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn is_prime(num: u32) -> bool {
    match num {
        0 | 1 => false,
        _ => {
            let limit = (num as f32).sqrt() as u32;

            !(2..=limit).any(|i| num % i == 0)
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn aura(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test_false() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(12))
    }

    #[test]
    fn simple_test_true() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(41));
    }
}
