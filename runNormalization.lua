require 'unsup'

dataDir = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/features/CQT_N_1024_P_4_O_4_R_24/'

trainData = torch.load(dataDir .. 'trainData.th')
validData = torch.load(dataDir .. 'validData.th')
testData = torch.load(dataDir .. 'testData.th')

function normalize(data)
	local nSamples = data:size(1)
	local nBands = data:size(2)
	local nFrames = data:size(3)
	
	for i = 1,nSamples do
		if (i % 100) == 0 then
			print(i .. '/' .. nSamples)
		end
		for j = 1,nFrames do
			local frame = data[{i,{},j}]
			frame:add(-torch.mean(frame))
			frame:div(math.max(0.001,torch.var(frame)))
		end
	end
end

norm(trainData)
norm(testData)
norm(validData)

			 






