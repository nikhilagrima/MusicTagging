-- Module which contains K separate linear + logSoftMax modules and a switch to select which one to use. This is useful when dealing with multiple labels.
-- When training, remember to update the parameter set every time you switch classifier. 

local MultiLabelLinear, parent = torch.class('nn.MultiLabelLinear','nn.Module')

function MultiLabelLinear:__init(nLabels, nInputs, nOutputs)
	parent.__init(self)
	self.switch = 1
	self.modules = {}
	for i=1,nLabels do
		local m = nn.Sequential()
		m:add(nn.Linear(nInputs,nOutputs))
		m:add(nn.LogSoftMax())
		table.insert(self.modules,m)
	end
end

function MultiLabelLinear:switch(s)
	self.switch = s
end

function MultiLabelLinear:updateOutput(input)
	return self.modules[self.switch]:updateOutput(input)
end

function MultiLabelLinear:updateGradInput(input,gradOutput)
	return self.modules[self.switch]:updateGradInput(input,gradOutput)
end

function MultiLabelLinear:accGradParameters(input, gradOutput, scale)
	self.modules[self.switch]:accGradParameters(input, gradOutput, scale)
end

function MultiLabelLinear:parameters()
	return self.modules[self.switch]:parameters()
end
