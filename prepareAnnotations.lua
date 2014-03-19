require 'csv'

------------------------------------------------------------------------------------
-- prepare all the data and annotations to be easily read by Torch
------------------------------------------------------------------------------------

--filepath = '/home/mikael/Work/Data/Magnatagatune/annotations_final.csv'
filepath = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/annotations_final.csv'

-- load data and partition into train/valid/test sets
query = csv.load{separator='\t',path=filepath,mode='query'}
training = query('union',{fold={0,1,2,3,4,5,6,7,8,9,'a','b'}})
valid = query('union',{fold={'c'}})
test = query('union',{fold={'d','e','f'}})

-- get the column names in the same order as the annotation file 
queryRaw = csv.load{separator='\t',path=filepath,mode='raw'}
colNames = queryRaw[1]
labelNames = {}
for i=1,#colNames-3 do
	table.insert(labelNames,colNames[i+1])
end

----------------------------------------
-- put all the labels in a torch tensor:
----------------------------------------


-- train
trlabels = {}
nTrain = #training.mp3_path
data = torch.Tensor(nTrain,#labelNames):zero()

for i=1,#labelNames do
	label = labelNames[i]
	for j=1,nTrain do
		data[j][i] = training[label][j]
	end
end

trlabels.nsamples=nTrain
trlabels.data=data
trlabels.labelNames=labelNames
trlabels.filePath=training.mp3_path
trlabels.clip_id=training.clip_id
trlabels.fold = training.fold


-- valid
vlabels = {}
nValid = #valid.mp3_path
data = torch.Tensor(nValid,#labelNames):zero()

for i=1,#labelNames do
	label = labelNames[i]
	for j=1,nValid do
		data[j][i] = valid[label][j]
	end
end

vlabels.nsamples=nValid
vlabels.data=data
vlabels.labelNames=labelNames
vlabels.filePath=valid.mp3_path
vlabels.clip_id=valid.clip_id
vlabels.fold = valid.fold


-- test
telabels = {}
nTest = #test.mp3_path
data = torch.Tensor(nTest,#labelNames):zero()

for i=1,#labelNames do
	label = labelNames[i]
	for j=1,nTest do
		data[j][i] = test[label][j]
	end
end

telabels.nsamples=nTest
telabels.data=data
telabels.labelNames=labelNames
telabels.filePath=test.mp3_path
telabels.clip_id=test.clip_id
telabels.fold = test.fold





-------------------------------------------------------------------------------------
-- Now we format all the data into files of the form [nSamples x nFreqs x nWindows]
-- this way we can straightforwardly apply Torch's convolution modules
-------------------------------------------------------------------------------------

-- these 3 songs have no data so we remove them
problems = {}
problems['6/norine_braun-now_and_zen-08-gently-117-146.th']=true
problems['8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.th']=true
problems['9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.th']=true


--dataDir = '/home/mikael/Work/Data/Magnatagatune/features/CQT_N_1024_P_4_O_4_R_24/'
dataDir = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/features/CQT_N_1024_P_4_O_4_R_24/'

-- training st
trainLabels=torch.Tensor(trlabels.nsamples-#problems,trlabels.data:size(2))
indx=1
for i=1,trlabels.nsamples do
	if (i % 100) == 0 then
		print(i .. '/ ' .. trlabels.nsamples)
	end
	file = trlabels.filePath[i]
	file = file:sub(1,#file-4) .. '.th'
	if not problems[file] then
		--print(file)
		x=torch.load(dataDir .. file)
		if not trainData then
			trainData = torch.FloatTensor(trlabels.nsamples,x:size(2),x:size(3))
		end
		trainData[{indx,{}}]:copy(x)
		trainLabels[{indx,{}}]:copy(trlabels.data[{i,{}}])
	else
		print(file .. ' is screwed up, skipping')
	end
	indx = indx + 1
end
max = torch.max(trainData)
min = torch.min(trainData)
trainData = (trainData-min)/(max-min)
torch.save(dataDir .. 'trainData.th',trainData)
torch.save(dataDir .. 'trainLabels.th',trainLabels)

-- validation set
validLabels=torch.Tensor(vlabels.nsamples,vlabels.data:size(2))
indx=1
for i=1,vlabels.nsamples do
	if (i % 100) == 0 then
		print('i / ' .. vlabels.nsamples)
	end
	file = vlabels.filePath[i]
	file = file:sub(1,#file-4) .. '.th'
	if not problems[file] then
		--print(file)
		x=torch.load(dataDir .. file)
		if not validData then
			validData = torch.FloatTensor(vlabels.nsamples,x:size(2),x:size(3))
		end
		validData[{indx,{}}]:copy(x)
		validLabels[{indx,{}}]:copy(vlabels.data[{i,{}}])
	else
		print(file .. ' is screwed up, skipping')
	end
	indx = indx + 1
end
validData = (validData-min)/(max-min)
torch.save(dataDir .. 'validData.th',validData)
torch.save(dataDir .. 'validLabels.th',validLabels)


-- test set
testLabels=torch.Tensor(telabels.nsamples,telabels.data:size(2))
indx=1
for i=1,telabels.nsamples do
	if (i % 100) == 0 then
		print('i / ' .. telabels.nsamples)
	end
	file = telabels.filePath[i]
	file = file:sub(1,#file-4) .. '.th'
	if not problems[file] then
		--print(file)
		x=torch.load(dataDir .. file)
		if not testData then
			testData = torch.FloatTensor(telabels.nsamples,x:size(2),x:size(3))
		end
		testData[{indx,{}}]:copy(x)
		testLabels[{indx,{}}]:copy(telabels.data[{i,{}}])
	else
		print(file .. ' is screwed up, skipping')
	end
	indx = indx + 1
end
testData = (testData-min)/(max-min)
torch.save(dataDir .. 'testData.th',testData)
torch.save(dataDir .. 'testLabels.th',testLabels)



