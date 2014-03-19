-- simple operations for complex numbers

complex = {}

function complex.new(x)
	local out=torch.zeros(x:nElement(),2)
	out[{{},1}]=x
	return out
end

function complex.abs(x)
	if x:nDimension() == 1 then
		return torch.abs(x)
	else
		return torch.norm(x,2,2)
	end
end

function complex.prod(x,y)
	local z=torch.Tensor(x:size())
	if y:nDimension() == 1 then
		z[{{},1}] = torch.cmul(x[{{},1}],y)
		z[{{},2}] = torch.cmul(x[{{},2}],y)
	else
		z[{{},1}] = torch.cmul(x[{{},1}],y[{{},1}]) - torch.cmul(x[{{},2}],y[{{},2}])
		z[{{},2}] = torch.cmul(x[{{},1}],y[{{},2}]) + torch.cmul(x[{{},2}],y[{{},1}])
	end
	return z
end
