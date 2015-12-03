--[[
	Copyright (c) 2016, Cristian Cardellino.
	This work is licensed under the "New BSD License". 
	See LICENSE for more information.
]]--

mltools.crossvalidation = {}

function mltools.crossvalidation.train_test_split(dataset, train_ratio)
    if dataset == nil or dataset.size == nil or dataset:size() <= 0 then
        error('Dataset should be a valid non-empty torch dataset')
    end
    if train_ratio == nil then train_ratio = 0.8 end

    local shuffle = torch.randperm(dataset:size())
    local train_size = math.floor(shuffle:size(1) * train_ratio)
    local test_size = shuffle:size(1) - train_size

    local train_dataset = {}
    local test_dataset = {}

    function train_dataset:size()
        return train_size
    end

    function test_dataset:size()
        return test_size
    end

    for i = 1, train_size do
        train_dataset[i] = {dataset[shuffle[i]][1]:clone(), dataset[shuffle[i]][2]}
    end

    for i = train_size + 1, train_size + test_size do
        test_dataset[i-train_size] = {dataset[shuffle[i]][1]:clone(), dataset[shuffle[i]][2]}
    end

    return train_dataset, test_dataset
end
