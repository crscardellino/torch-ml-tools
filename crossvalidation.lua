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
    local features_size = dataset[1][1]:size()[1]

    local X_train = torch.Tensor(train_size, features_size)
    local X_test = torch.Tensor(test_size, features_size)
    local y_train = torch.Tensor(train_size)
    local y_test = torch.Tensor(test_size)

    for i = 1, train_size do
        X_train[i] = dataset[shuffle[i]][1]:clone()
        y_train[i] = dataset[shuffle[i]][2]
    end

    for i = train_size + 1, train_size + test_size do
        X_test[i-train_size] = dataset[shuffle[i]][1]:clone()
        y_test[i-train_size] = dataset[shuffle[i]][2]
    end

    return X_train, X_test, y_train, y_test
end
