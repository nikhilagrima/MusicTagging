-- TODO: finish/test

dataPath = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/features/CQT_N_1024_P_4_O_4_R_24/'
trdata = torch.load(dataPath .. 'trainData.th')
nSamples = trdata:size(1)
nFeatures = trdata:size(2)

-- compute covariance matrix
cov = trdata:t() * trdata
cov = cov / nSamples

-- compute distance
dist = torch.Tensor(nFeatures,nFeatures)

for i=1,nFeatures do
	for j=1,nFeatures do
		local xi = trdata[{{},i}] / math.sqrt(cov(i,i))
		local xj = trdata[{{},j}] / math.sqrt(cov(j,j))
		dist(i,j) = torch.mean(torch.cmul(xi - xj, xi - xj))
	end
end

W = torch.exp(-dist/(2*sigma))
D = torch.Tensor(nFeatures,nFeatures)
for i=1,nFeatures do
	D[i][i] = 1/math.sqrt(torch.sum(W[{i,{}}]))
end
L = torch.eye(nFeatures) - D * W * D

