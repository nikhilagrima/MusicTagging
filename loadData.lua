-----------------------------------------------------------------------------
-- general function to load different subsets of the data/labels
-----------------------------------------------------------------------------

function loadData(dataPath, options)
	if options.size == 'full'	then
		if options.preproc == 'PCA' then
			trainData = torch.load(dataPath .. 'trainDataPCA.th').trainData
			validData = torch.load(dataPath .. 'validDataPCA.th').validData
			testData = torch.load(dataPath .. 'testDataPCA.th').testData
		elseif options.preproc == 'ZCA' then
			trainData = torch.load(dataPath .. 'trainDataZCA.th').trainData
			validData = torch.load(dataPath .. 'validDataZCA.th').validData
			testData = torch.load(dataPath .. 'testDataZCA.th').testData
		else
			trainData = torch.load(dataPath .. 'trainData.th')
			validData = torch.load(dataPath .. 'validData.th')
			testData = torch.load(dataPath .. 'testData.th')
		end
		trainLabels = torch.load(dataPath .. 'trainLabels.th')
		validLabels = torch.load(dataPath .. 'validLabels.th')
		testLabels = torch.load(dataPath .. 'testLabels.th')
	elseif options.size == 'medium'	then
		trainData = torch.load(dataPath .. 'testData.th')
		trainLabels = torch.load(dataPath .. 'testLabels.th')
		testData = torch.load(dataPath .. 'validData.th')
		testLabels = torch.load(dataPath .. 'validLabels.th')
	elseif options.size == 'toy' then	
		x=torch.load('toydata2.th')
		trainData=x.trdata
		testData=x.tedata
		labels = torch.load(dataPath .. 'validLabels.th')
		trainLabels = labels[{{1,1000},{}}] 
		testLabels = labels[{{1001,1800},{}}] 
	end

	if not (options.size == 'toy') then
		trainData = trainData:resize(trainData:size(1),trainData:size(2),1,trainData:size(3))
		testData = testData:resize(testData:size(1),testData:size(2),1,testData:size(3))
		if validData then
			validData = validData:resize(validData:size(1),validData:size(2),1,validData:size(3))
		end
	end

	if options.norm == 'sphere' then
		function norm(data)
			local nSamples = data:size(1)
			local nFrames = data:size(4)
			for i=1,nSamples do
				if (i % 100) == 0 then
					print(i .. '/' .. nSamples)
				end
				for j=1,nFrames do
					local d = data[{i,{},{},j}]
					d:add(-torch.mean(d))
					d:div(math.max(0.01,torch.var(d)))
				end
			end
		end
		norm(trainData)
		norm(testData)
		if validData then
			norm(validData)
		end
	end

	if options.labels == 'all' then
		lab = {}
		labelNames=torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/labelNames.th')
	elseif options.labels == 'few' then
		lab = {34,38}
		labelNames=torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/labelNames.th')
	elseif options.labels == 'one' then
		lab = {34,34}
		labelNames=torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/labelNames.th')
	elseif options.labels == 'top50' then
		local nLabels = 50
		local counts = torch.sum(trainLabels,1)
		_,indx = torch.sort(-counts)
		indx = indx[1]
		trainLabels2 = torch.Tensor(trainLabels:size(1),nLabels):zero()
		testLabels2 = torch.Tensor(testLabels:size(1),nLabels):zero()
		validLabels2 = torch.Tensor(validLabels:size(1),nLabels):zero()
		labelNames=torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/labelNames.th')
		labelNames2 = {}
		for i=1,nLabels do
			trainLabels2[{{},i}]:copy(trainLabels[{{},indx[i]}])
			testLabels2[{{},i}]:copy(testLabels[{{},indx[i]}])
			validLabels2[{{},i}]:copy(validLabels[{{},indx[i]}])
			labelNames2[i] = labelNames[indx[i]]
		end
		trainLabels = trainLabels2
		testLabels = testLabels2
		validLabels = validLabels2
		labelNames = labelNames2	
	end

	if options.labels ~= 'top50' then
		trainLabels = trainLabels[{{},lab}]
		testLabels = testLabels[{{},lab}]
		if validLabels then
			validLabels = validLabels[{{},lab}]
		end
	end

	data = {}
	data.train = trainData
	data.test = testData
	data.valid = validData
	labels = {}
	labels.train = trainLabels + 1
	labels.test = testLabels + 1
	if validLabels then
		labels.valid = validLabels + 1
	end
	labels.names = labelNames

	return data, labels
end
	
