--------------------------------------------------------------------
-- preprocess the data in various ways and save
--------------------------------------------------------------------
require 'unsup'

dataDir = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/features/CQT_N_1024_P_4_O_4_R_24/'

trainData = torch.load(dataDir .. 'trainData.th')
validData = torch.load(dataDir .. 'validData.th')
testData = torch.load(dataDir .. 'testData.th')

nTrainSamples = trainData:size(1)
nTestSamples = testData:size(1)
nValidSamples = validData:size(1)
nBands = trainData:size(2)
nFrames = trainData:size(3)

trainData = trainData:transpose(2,3):resize(nTrainSamples*nFrames,nBands)
validData = validData:transpose(2,3):resize(nValidSamples*nFrames,nBands)
testData = testData:transpose(2,3):resize(nTestSamples*nFrames,nBands)

-- center the data
means = torch.mean(trainData,1)
trainData:add(-torch.expand(means:resize(1,nBands),trainData:size(1),nBands))
validData:add(-torch.expand(means:resize(1,nBands),validData:size(1),nBands))
testData:add(-torch.expand(means:resize(1,nBands),testData:size(1),nBands))

-- compute covariance matrix
cov = trainData:t() * trainData
ce,cv = torch.symeig(cov,'V')
ce:add(1e-5):sqrt()
invce = ce:clone():pow(-1)
invdiag = torch.diag(invce)

-- PCA whitening
print('Applying PCA whitening to:')
print('training data')
trainDataPCA, means, P, invP = unsup.pca_whiten(trainData)
print('validation data')
validDataPCA = unsup.pca_whiten(validData, means, P, invP)
print('testing data')
testDataPCA = unsup.pca_whiten(testData, means, P, invP)


-- ZCA whitening
print('Applying ZCA whitening to:')
print('training data')
trainDataZCA, means, Z, invZ = unsup.zca_whiten(trainData)
print('validation data')
validDataPCA = unsup.zca_whiten(validData, means, Z, invZ)
print('testing data')
testDataPCA = unsup.zca_whiten(testData, means, Z, invZ)


-- resize everything 
trainDataPCA = trainDataPCA:resize(nTrainSamples, nFrames, nBands):transpose(2,3)
validDataPCA = validDataPCA:resize(nValidSamples, nFrames, nBands):transpose(2,3)
testDataPCA = testDataPCA:resize(nTestSamples, nFrames, nBands):transpose(2,3)

trainDataZCA = trainDataZCA:resize(nTrainSamples, nFrames, nBands):transpose(2,3)
validDataZCA = validDataZCA:resize(nValidSamples, nFrames, nBands):transpose(2,3)
testDataZCA = testDataZCA:resize(nTestSamples, nFrames, nBands):transpose(2,3)

-- clone to make contiguous
trainDataPCA = trainDataPCA:clone()
validDataPCA = validDataPCA:clone()
testDataPCA = testDataPCA:clone()

trainDataZCA = trainDataZCA:clone()
validDataZCA = validDataZCA:clone()
testDataZCA = testDataZCA:clone()

-- save
torch.save(dataDir .. 'trainDataPCA.th',{trainData=trainDataPCA,means=means,P=P,Pinv=Pinv})
torch.save(dataDir .. 'testDataPCA.th',{testData=testDataPCA,means=means,P=P,Pinv=Pinv})
torch.save(dataDir .. 'validDataPCA.th',{validData=validDataPCA,means=means,P=P,Pinv=Pinv})

torch.save(dataDir .. 'trainDataZCA.th',{trainData=trainDataZCA,means=means,Z=Z,Zinv=Zinv})
torch.save(dataDir .. 'testDataZCA.th',{testData=testDataZCA,means=means,Z=Z,Zinv=Zinv})
torch.save(dataDir .. 'validDataZCA.th',{validData=validDataZCA,means=means,Z=Z,Zinv=Zinv})







