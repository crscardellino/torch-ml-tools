package = "MLTools"
version = "0.1-1"

source = {
   url = "git://github.com/crscardellino/torch-ml-tools.git",
   tag = "0.1-1"
}

description = {
   summary = "Torch Machine Learning Tools",
   detailed = [[
   	    A set of Machine Learning tools useful for the Torch
   	    Framework, inspired by Python's Scikit-Learn
   ]],
   homepage = "https://github.com/crscardellino/torch-ml-tools"
}

dependencies = {
   "torch >= 7.0"
}

build = {
	type = "builtin",
	modules = {
		['mltools.init'] = 'init.lua',
		['mltools.SparseDataset'] = 'SparseDataset.lua',
		['mltools.crossvalidation'] = 'crossvalidation.lua'
	}
}
