def unpack_args(f):
    return lambda x: f(*x)


class Input:
    def __init__(self, index):
        self.index = index

    def compute(self, input_values):
        return input_values[self.index]


class LazyVal:
    def __init__(self, func, inputs):
        """Inputs must be of type Input or LazyVal"""
        self.func = func
        self.inputs = inputs

    def compute(self, input_values):
        computed_inputs = [input.compute(input_values) for input in self.inputs]
        return unpack_args(self.func)(computed_inputs)


class DAG:
    def __init__(self, executor):
        self.executor = executor
        self.num_partitions = 0
        self.partitioned_inputs = []

    def add_input(self, partitioned_input):
        assert len(partitioned_input) > 0
        if self.num_partitions == 0:
            self.num_partitions = len(partitioned_input)
        else:  # further inputs must have same number of partitions as first
            assert self.num_partitions == len(partitioned_input)
        index = len(self.partitioned_inputs)
        self.partitioned_inputs.append(partitioned_input)
        return Input(index)

    def transform(self, func, inputs):
        assert len(inputs) > 0
        return LazyVal(func, inputs)

    def _get_zipped_inputs(self):
        # iterate over inputs one partition at a time
        return zip(*self.partitioned_inputs)

    def compute(self, output):
        return self.executor.map(output.compute, self._get_zipped_inputs())
