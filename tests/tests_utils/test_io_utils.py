import pytest

from hyperparameter_searcher.utils.io_utils import debug_function, fullname


class TestClass:
    pass


@pytest.mark.parametrize(["x", "y"], [(2, 4)])
def test_debug_function(x, y):
    assert debug_function(x) == y


@pytest.mark.parametrize(
    ["cls", "full_name"], [(TestClass, "tests.tests_utils.test_io_utils.TestClass")]
)
def test_fullname(cls, full_name):
    assert fullname(cls) == full_name
