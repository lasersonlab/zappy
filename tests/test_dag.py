import concurrent.futures
import pytest
from zappy.executor.dag import DAG


def add_one(x):
    return x + 1


def times_two(x):
    return x * 2


def add(x, y):
    return x + y


def test_dag_single_partition():
    dag = DAG(concurrent.futures.ThreadPoolExecutor())
    input = dag.add_input([2])
    output = dag.transform(add_one, [input])
    assert list(dag.compute(output)) == [3]


def test_dag_single_partition_serial_functions():
    dag = DAG(concurrent.futures.ThreadPoolExecutor())
    input = dag.add_input([2])
    intermediate = dag.transform(add_one, [input])
    output = dag.transform(times_two, [intermediate])
    assert list(dag.compute(output)) == [6]


def test_dag_single_partition_binary_function():
    dag = DAG(concurrent.futures.ThreadPoolExecutor())
    input1 = dag.add_input([2])
    input2 = dag.add_input([3])
    output = dag.transform(add, [input1, input2])
    assert list(dag.compute(output)) == [5]


def test_dag_multiple_partitions():
    dag = DAG(concurrent.futures.ThreadPoolExecutor())
    input = dag.add_input([2, 3, 5])
    output = dag.transform(add_one, [input])
    assert list(dag.compute(output)) == [3, 4, 6]


def test_incompatible_num_partitions():
    dag = DAG(concurrent.futures.ThreadPoolExecutor())
    dag.add_input([2])
    with pytest.raises(AssertionError):
        dag.add_input([1, 5])


def test_no_transform():
    dag = DAG(concurrent.futures.ThreadPoolExecutor())
    output = dag.add_input([2])
    assert list(dag.compute(output)) == [2]
