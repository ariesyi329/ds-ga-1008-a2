require 'xlua'
require 'image'
require 'torch'
require 'optim'
require 'cunn'
require 'nn'
dofile './provider.lua'

function parseDataLabel(d, numSamples, numChannels, height, width)
	local t = torch.ByteTensor(numSamples, numChannels, height, width)
	--local l = torch.ByteTensor(numSamples)
	local idx = 1
	for i = 1, numSamples do
		local this_d = d[i]
		--for j = 1, #this_d do
		t[i]:copy(this_d)
		--l[idx] = i
		idx = idx + 1
	end
	assert(idx == numSamples+1)
	return t
end

--local trainSize = 4000
local testSize = 30000
local channel = 3
local height = 96
local width = 96

--local model_file = "logs/vgg/baseline_epoch_230.net"
local model_file = "logs/augments/model.net"
-- local model_file = "logs/kmeans3/model.net"

local rawTest = torch.load('stl-10/extra.t7b')
--provider = torch.load("provider.t7")


testData = {
	data = torch.Tensor(),
	--labels = torch.Tensor(),
	size = function() return testSize end
}

selectedData = {
	data = torch.Tensor(),
	labels = torch.Tensor()
}

print "==> loading test data..."

testData.data = parseDataLabel(rawTest.data[1], testSize, channel, height, width)

-- convert from ByteTensor to Float
testData.data = testData.data:float()
collectgarbage()

-- Load the model

print "==> loading model..."
model = torch.load(model_file)

print "==> predicting..."
model:cuda()
model:evaluate()


local bs = 25
local currIdx = 1
local selectedNum = 1
for t = 1,testData.size(1),bs do

	xlua.progress(t, testData.size())

	local outputs = model:forward(testData.data:narrow(1,t,bs):cuda())
	--confusion:batchAdd(outputs, testData.labels:narrow(1,t,bs))

	for k = 1, bs do
		local label = torch.LongTensor()
		local _max = torch.FloatTensor()
		_max:max(label, outputs[k]:float(), 1)
      if torch.max(outputs[k]:float()) > 12 then
        	selectedData.data[selectedNum] = testData.data[currIdx]
 			selectedData.labels[selectedNum] = label[1]
			selectedNum = selectedNum+1
		currIdx = currIdx + 1
	end
end



torch.save('selected1.t7', selectedData)
end
