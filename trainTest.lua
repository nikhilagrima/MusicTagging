-----------------------------------------------------------
-- Generic function to train a model for one epoch
-----------------------------------------------------------
--
dofile('utils.lua')
require 'image'

function trainModel(model, criterion, trdata, trlabels, tedata, telabels, opt, logger)
	local optimMethod = opt.optimMethod
	local optimState = opt.optimState
	local batchSize = opt.batchSize
	local trsize = trdata:size(1)
	local tesize = tedata:size(1)
	local nLabels = trlabels:size(2)
	local confusion = {}
	for j = 1,nLabels do
		confusion[j] = optim.ConfusionMatrix(opt.classes)
	end
	local inputs
	local targets
	local batchSize = 32
	
	local labelIndx = {}
	for i=1,nLabels do
		labelIndx[i] = {}
		-- get the indices of positive and negative instances
		labelIndx[i].pos = torch.find(torch.eq(trlabels[{{},i}],2))
		labelIndx[i].neg = torch.find(torch.eq(trlabels[{{},i}],1))
		-- indices for shuffling
		labelIndx[i].posShuffle = torch.randperm(labelIndx[i].pos:size(1))
		labelIndx[i].negShuffle = torch.randperm(labelIndx[i].neg:size(1))
		-- pointers for current indices
		labelIndx[i].posIndx = 1
		labelIndx[i].negIndx = 1
	end

	if opt.type == 'cuda' then
		inputs = torch.CudaTensor(batchSize, trdata:size(2), 1, trdata:size(4))
		targets = torch.CudaTensor(batchSize)
	else
		inputs = torch.Tensor(batchSize, trdata:size(2), 1, trdata:size(4))
		targets = torch.Tensor(batchSize)
	end
	local outputs = torch.Tensor(batchSize, nOutputs)

	-- get model parameters. We store pointers to the parameter set
	-- for each label, so that we don't have to keep moving stuff from
	-- the GPU.
	local w = {}
	local dL_dw = {}
	for j = 1,nLabels do
		model.modules[#model.modules].switch = j
		w[j],dL_dw[j] = model:getParameters()
	end

	-- vector to store the loss for each label
	local losses = torch.Tensor(nLabels):zero()

	-- we define one epoch to be trsize/batchsize loops over the labels.
	for t = 1,(trsize-batchSize),batchSize do
		xlua.progress(t,trsize)
		-- loop over labels
		for j = 1,nLabels do
		
			-- make balanced minibatch for this label
			for i=1,batchSize,2 do
				local a = labelIndx[j]
				local indxPos = a.pos[a.posShuffle[a.posIndx]]
				a.posIndx = a.posIndx % a.pos:size(1) + 1
				local indxNeg = a.neg[a.negShuffle[a.negIndx]]
				a.negIndx = a.negIndx % a.neg:size(1) + 1
				--print(a.posIndx .. '/' .. a.negIndx)
				inputs[{i,{}}]:copy(trdata[indxPos])
				inputs[{i+1,{}}]:copy(trdata[indxNeg])
				targets[i] = trlabels[{indxPos,j}]
				targets[i+1] = trlabels[{indxNeg,j}]
	
				if (not trlabels[{indxPos,j}] == 1) or (not trlabels[{indxNeg,j}] == 0) then
					error('shit')
				end

			end

			-- set the output classifier to the current label
			model.modules[#model.modules].switch=j	

			-- create closure to evaluate L(w) and dL/dw
			local feval = function(w_)
				
				if w_ ~= w[j] then
					error('weird')
					w[j]:copy(w_)
				end

				-- reset gradients
				dL_dw[j]:zero()
				
				-- L is the average loss
				local L = 0
				
				local outputs = model:forward(inputs)
				L = criterion:forward(outputs,targets)
				local dL_do = criterion:backward(outputs,targets)
				model:backward(inputs,dL_do)
				for k = 1,batchSize do
					confusion[j]:add(outputs[k],targets[k])
				end
		
				L = L/batchSize
				-- return L and dL/dw
				return L, dL_dw[j]
			end
			-- optimize on current mini-batch
			local _, batchLoss = optimMethod(feval,w[j],optimState)
			losses[j] = losses[j] + batchLoss[1]
		end
	end

	inputs:zero()
	targets:zero()

	---------------------------------------------
	-- compute the mean AUC
	---------------------------------------------
	
	local aucTrain = torch.Tensor(nLabels):zero()
	local aucTest = torch.Tensor(nLabels):zero()	

	-- TODO: account for remaining (trsize % batchSize) samples
	local pred = torch.Tensor(trsize-(trsize % batchSize),nLabels)
	local fextractor = model:clone()
	fextractor.modules[#fextractor.modules] = nil
	local classifier = model.modules[#model.modules]:clone()
	-- AUC on training set
	for k = 1,(trsize-batchSize),batchSize do
		inputs:copy(trdata[{{k, k + batchSize - 1},{}}])
		local features = fextractor:forward(inputs)
		for j = 1,nLabels do
			classifier.switch = j
			local p = classifier:forward(features)
			pred[{{k, k + batchSize - 1},j}]:copy(p[{{},2}])
		end
	end
	for j = 1,nLabels do
		aucTrain[j] = auc(trlabels[{{1,trsize-(trsize % batchSize)},j}]-1,pred[{{},j}])
	end

	-- AUC on testing set
	pred = torch.Tensor(tesize-(tesize % batchSize),nLabels)
	for k = 1,(tesize-batchSize),batchSize do
		inputs:copy(tedata[{{k, k + batchSize - 1},{}}])
		local features = fextractor:forward(inputs)
		for j = 1,nLabels do
			classifier.switch = j
			local p = classifier:forward(features)
			pred[{{k, k + batchSize - 1},j}]:copy(p[{{},2}])
		end
	end
	for j = 1,nLabels do
		aucTest[j] = auc(telabels[{{1,tesize-(tesize % batchSize)},j}]-1,pred[{{},j}])
	end
	
	print('Loss   /   AUC (train)   /   AUC (test): ')
	local a = torch.Tensor(nLabels,3)
	a[{{},1}]:copy(losses)
	a[{{},2}]:copy(aucTrain)
	a[{{},3}]:copy(aucTest)
	print(a)
	print('Mean AUC (train set): ' .. torch.mean(aucTrain))
	print('Mean AUC (test set) : ' .. torch.mean(aucTest))
	-- record log info
	logger:add{['mean AUC (train set)'] = torch.mean(aucTrain), ['mean AUC (test set)'] = torch.mean(aucTest)}
	logger:style{['mean AUC (train set)'] = '-', ['mean AUC (test set)'] = '-'}
	logger:plot()	
end


			































