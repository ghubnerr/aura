import pytest
import aura


def test_sum_as_string():
    assert aura.sum_as_string(1, 1) == "2"
