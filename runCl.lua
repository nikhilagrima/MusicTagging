-- general function for classification

require 'optim'
require 'nn'
dofile('trainTest.lua')
dofile('MultiLabelLinear.lua')
dofile('loadData.lua')

-- process command line options
cmd = torch.CmdLine()
cmd:option('-model',1,' type of model to construct')
cmd:option('-batchSize',1, 'mini-batch size')
cmd:option('-epochs',5)
cmd:option('-type','float','type: double | float | cuda')
cmd:option('-save','Results/')
cmd:option('-gpunum',1)
cmd:option('-norm','none')
cmd:option('-weightDecay',0)
cmd:option('-plot',true)
cmd:option('-nOutputPlanes',32)
cmd:option('-kernelSize',1)
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
	require 'cunn'
	cutorch.setDevice(opt.gpunum)
elseif opt.type == 'float' then
	torch.setdefaulttensortype('torch.FloatTensor')
end


opt.save = opt.save .. '/LR_0.5/' .. 'model_' .. opt.model .. '/kernelSize_' .. opt.kernelSize .. '_nOutputPlanes_' .. opt.nOutputPlanes .. '/weightDecay_' .. opt.weightDecay .. '/'


--------------------------------------------------
-- LOAD DATASET
--------------------------------------------------

-- CQT parameters
N = 1024
P = 4
O = 4
R = 24

dataPath = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/features/CQT_N_' .. N .. '_P_' .. P .. '_O_' .. O .. '_R_' .. R .. '/'
--dataPath = '/home/mikael/Work/Data/Magnatagatune/features/CQT_N_1024_P_4_O_4_R_24/'


data, labels = loadData(dataPath, {size='full',labels='top50', norm=opt.norm})
trdata = data.train
tedata = data.valid
trlabels = labels.train
telabels = labels.valid


nBands = trdata:size(2)
nFrames = trdata:size(4)


------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------


nOutputs = 2

if opt.model == 1 then
	-- toy model: pooling over frames -> linear classifier
	model = nn.Sequential()
	--model:add(nn.SpatialMaxPooling(nFrames,1,1,1))
	model:add(nn.SpatialLPPooling(nBands,2,nFrames,1,nFrames,1))
	model:add(nn.Reshape(nBands))
	model:add(nn.MultiLabelLinear(trlabels:size(2),nBands,2))
elseif opt.model == 2 then
	poolsize = 100
	poolstride = poolsize
	nOutputPlanes = opt.nOutputPlanes
	kernelSize = opt.kernelSize
	paddedSize = 2000


	model = nn.Sequential()
	model:add(nn.SpatialZeroPadding(0,paddedSize-nFrames,0,0))
	model:add(nn.SpatialConvolutionBatch(nBands,nOutputPlanes,kernelSize,1,1,1))
	model:add(nn.Threshold())
	model:add(nn.SpatialMaxPooling(poolsize,1,poolstride,1))
	model:add(nn.Reshape(nOutputPlanes * math.floor((paddedSize-kernelSize+1)/poolsize)))
	model:add(nn.MultiLabelLinear(trlabels:size(2),nOutputPlanes * math.floor((paddedSize-kernelSize+1)/poolsize),2))
end

-- loss
criterion = nn.ClassNLLCriterion()

print(model)
-----------------------------------------
-- OPTIMIZATION
-----------------------------------------

optimMethod = optim.sgd
optimState = {
	learningRate = 0.2,
	weightDecay = opt.weightDecay,
	learningRateDecay = 1e-7
}

opt.optimState = optimState
opt.optimMethod = optimMethod
opt.classes = {'1','2'}



-------------------------------------------
-- TRAIN
-------------------------------------------

if opt.type == 'float' then
	model = model:float()
elseif opt.type == 'double' then
	model = model:double()
elseif opt.type == 'cuda' then
	model = model:cuda()
	criterion = criterion:cuda()
end

-- create logger
logger = optim.Logger(opt.save .. '/logger.log')

torch.manualSeed(123)

for i = 1,opt.epochs do
	-- train
	trainModel(model, criterion, trdata, trlabels, tedata, telabels, opt, logger)
	
	-- save/log current model
	local filename = paths.concat(opt.save, 'model_epoch_' .. i .. '.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	print('==> saving model to ' .. filename)
	--model:float()
	torch.save(filename, {model = model, logger = logger})
	-- delete previous model to save space
	if i > 1 then
		os.execute('rm ' .. paths.concat(opt.save, 'model_epoch_' .. i-1 .. '.net'))
	end
end


















