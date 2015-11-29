require 'mltools'

local empty_dataset = mltools.SparseDataset()

assert(empty_dataset.indices == nil)
assert(empty_dataset.values == nil)
assert(empty_dataset.target == nil)
assert(empty_dataset.shape == nil)