import concurrent.futures


def unpack_args(f):
    return lambda x: f(*x)


class Input:
    def __init__(self, index):
        self.index = index

    def compute(self, input_values):
        return input_values[self.index]

    def node(self):
        return '"Input({}, {})"'.format(id(self), self.index)

    def dot(self, s=""):
        s += "{}\n".format(self.node())
        return s


class LazyVal:
    def __init__(self, func, inputs):
        """Inputs must be of type Input or LazyVal"""
        self.func = func
        self.inputs = inputs

    def compute(self, input_values):
        computed_inputs = [input.compute(input_values) for input in self.inputs]
        return unpack_args(self.func)(computed_inputs)

    def node(self):
        return '"LazyVal({}, {})"'.format(id(self), self.func)

    def dot(self, s=""):
        for input in self.inputs:
            s += "{} -> {}\n".format(self.node(), input.node())
        for input in self.inputs:
            s = input.dot(s)
        return s


class DAG:
    def __init__(self, executor):
        self.executor = executor
        self.local_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.num_partitions = 0
        self.partitioned_inputs = []

    def add_input(self, partitioned_input):
        assert len(partitioned_input) > 0
        # first input sets the number of partitions
        if self.num_partitions == 0:
            self.num_partitions = len(partitioned_input)
        # further inputs cannot grow the number of partitions
        elif self.num_partitions < len(partitioned_input):
            assert self.num_partitions >= len(partitioned_input)
        # but they can shrink (e.g. when subsetting rows)
        else:
            self.num_partitions = len(partitioned_input)
            self.partitioned_inputs = [
                inputs[: self.num_partitions] for inputs in self.partitioned_inputs
            ]
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
        # Uncomment to see the dot representation of the DAG
        # print("digraph compute {{\n{}}}".format(output.dot()))
        if len(list(self._get_zipped_inputs())) == 1:
            # run single partitions locally
            return self.local_executor.map(output.compute, self._get_zipped_inputs())
        return self.executor.map(output.compute, self._get_zipped_inputs())

    def compute_multiple(self, outputs):
        def apply_multiple(x):
            return tuple(f(x) for f in tuple(output.compute for output in outputs))

        return zip(*self.executor.map(apply_multiple, self._get_zipped_inputs()))
