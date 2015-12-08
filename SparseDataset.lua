local SparseDataset = torch.class('mltools.SparseDataset')

local function torchTensor(tensor_type)
    local tensor_types = {
        ['torch.ByteTensor'] = torch.ByteTensor,
        ['torch.CharTensor'] = torch.CharTensor,
        ['torch.ShortTensor'] = torch.ShortTensor,
        ['torch.IntTensor'] = torch.IntTensor,
        ['torch.LongTensor'] = torch.LongTensor,
        ['torch.FloatTensor'] = torch.FloatTensor,
        ['torch.DoubleTensor'] = torch.DoubleTensor,
        ['torch.CudaTensor'] = torch.CudaTensor
    }

    return tensor_types[tensor_type]
end

function SparseDataset:__init(indices, values, target, shape, num_classes, data_type, target_type)
    if type(indices) == 'string' then
        local fname = indices
        local zero_based = shape

        self.data_type = values or 'torch.DoubleTensor'
        self.target_type = target or 'torch.IntTensor'

        if not torchTensor(self.data_type) or not torchTensor(self.target_type) then
            error('The data_type and target_type should be a valid torch tensor type')
        end

        if not zero_based then zero_based = false end
        if type(zero_based) ~= 'boolean' then
            error('Wrong zero_based argument. Should be of type boolean.')
        end -- Checks if the indices starts with 0 or 1 (default)

        self:readFromFile(fname, zero_based)
    elseif type(indices) == 'table' then
        if type(values) ~= 'table' or not target or type(shape) ~= 'table'
                or type(num_classes) ~= 'number' then
            error('Wrong argument type')
        end

        self.data_type = data_type or 'torch.DoubleTensor'
        self.target_type = target_type or 'torch.IntTensor'
        if not torchTensor(self.data_type) or not torchTensor(self.target_type) then
            error('The indices_type, data_type and target_type, should be a valid torch tensor type')
        end

        self.indices = indices  -- Table of Tensors
        self.values = values  -- Table of Tensors
        self.target = target:type(self.target_type)  -- One dimensional Tensor
        self.shape = shape  -- Tuple with 2 values: rows and cols
        self.num_classes = num_classes

        if self.target:min() == 0 then self.target = self.target + 1 end  -- Make sure the target classes start at 1
    end
end

function SparseDataset:getDenseDataset()
    local dataset = torchTensor(self.data_type)(self.shape.rows, self.shape.cols):zero()

    for i = 1, #self.indices do
        dataset[{i, {}}]:indexAdd(1,
            self.indices[i]:type('torch.LongTensor'),
            self.values[i]:type(self.data_type)
        )
    end

    return dataset
end

function SparseDataset:getTorchDataset()
    local dataset = {}
    local datasetSize = self.shape.rows
    local denseDataset = self:getDenseDataset()

    function dataset:size()
        return datasetSize
    end

    for i = 1, dataset:size() do
        dataset[i] = {denseDataset[i], self.target[i]}
    end

    return dataset
end

function SparseDataset:readFromFile(fname, zero_based)
    if zero_based == nil then zero_based = false end

    print('Reading ' .. fname)

    local function readline(line)
        local label = tonumber(string.match(line, '^([%+%-]?%s?%d+)'))
        if not label then
            error('could not read label')
        end
        local vals = {}
        local inds = {}
        local indcntr = 0
        for ind, val in string.gmatch(line, '(%d+):([%+%-]?%d?%.?%d+)') do
            indcntr = indcntr + 1
            ind = tonumber(ind)
            val = tonumber(val)
            if not ind or not val then
                error('reading failed')
            end
            if zero_based then ind = ind + 1 end  -- We transform the zero based in one-based
            if ind < indcntr then
                error('indices are not in increasing order')
            end
            inds[#inds+1] = ind
            vals[#vals+1] = val
        end
        return torch.LongTensor(inds), torchTensor(self.data_type)(vals), label
    end

    local indices = {}
    local values = {}
    local target = {}
    local shape = {rows = 0, cols = 0}
    local labels = {}
    setmetatable(labels, { __index = function() return 0 end })
    for line in io.lines(fname) do
        local inds, vals, tgt = readline(line)

        indices[#indices+1] = inds
        values[#values+1] = vals
        target[#target+1] = tgt

        shape.rows = shape.rows + 1
        shape.cols = math.max(shape.cols, inds[-1])

        labels[tgt] = labels[tgt] + 1
    end

    local shift = 0

    if labels[shift] then shift = 1 end

    local num_classes = 0

    for l, c in pairs(labels) do
        io.write(string.format("# of samples of label %d = %d\n", l + shift, c))
        num_classes = num_classes + 1
    end

    io.write(string.format("# of total samples = %d\n", shape.rows))
    io.write(string.format("# of features = %d\n", shape.cols))

    self.indices = indices
    self.values = values
    self.target = torchTensor(self.target_type)(target)
    self.shape = shape
    self.num_classes = num_classes

    if self.target:min() == 0 then self.target = self.target + 1 end  -- Make sure the target classes start at 1
end

function SparseDataset:size()
    return torch.IntTensor({self.shape.rows, self.shape.cols})
end

function SparseDataset:writeToFile(fname)
    print('Writing ' .. fname)

    local function vectostr(i, x)
        local str = {}
        local cntr = 1
        x:apply(function(v)
            table.insert(str, string.format('%d:%g', i[cntr], v))
            cntr = cntr + 1
            return v
        end)
        return table.concat(str, ' ')
    end

    local of = torch.DiskFile(fname, 'w')
    for i = 1, #self.indices do
        of:writeString(string.format('%g %s\n', self.target[i], vectostr(self.indices[i], self.values[i])))
    end
    of:close()
end