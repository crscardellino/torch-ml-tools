--[[
	Copyright (c) 2016, Cristian Cardellino.
	This work is licensed under the "New BSD License".
	See LICENSE for more information.
]]--

local SparseDataset = torch.class('mltools.SparseDataset')

function SparseDataset:__init(fname_or_indices, zero_based_or_values, target, shape)
    if type(fname_or_indices) == 'string' then
        if zero_based_or_values == nil then zero_based_or_values = false end
        if type(zero_based_or_values) ~= 'boolean' then error('Wrong second argument. Should be of type boolean.') end
        -- If the indices starts with 0 or 1 (default)
        self:readFromFile(fname_or_indices, zero_based_or_values)
    elseif type(fname_or_indices) == 'table' then
        if type(zero_based_or_values) ~= 'table' or not target or type(shape) ~= 'table' then
            error('Wrong argument type')
        end

        self.indices = fname_or_indices  -- Table of Tensors
        self.values = zero_based_or_values  -- Table of Tensors
        self.target = target:type('torch.IntTensor')  -- One dimensional Tensor
        self.shape = shape  -- Tuple with 2 values: rows and cols
    end
end

function SparseDataset:getDenseDataset()
    local dataset = torch.FloatTensor(self.shape.rows, self.shape.cols):zero()

    for i = 1, #self.indices do
        dataset[{i, {}}]:indexAdd(1,
            self.indices[i]:type('torch.LongTensor'),
            self.values[i]:type('torch.FloatTensor')
        )
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
        return torch.LongTensor(inds), torch.FloatTensor(vals), label
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

    for l, c in pairs(labels) do
        io.write(string.format("# of samples of label %d = %d\n", l, c))
    end
    io.write(string.format("# of total samples = %d\n", shape.rows))
    io.write(string.format("# of features = %d\n", shape.cols))

    self.indices = indices
    self.values = values
    self.target = torch.IntTensor(target)
    self.shape = shape
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