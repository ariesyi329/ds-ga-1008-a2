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

--[[
print "==> normalize testData"
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,testData:size() do
	xlua.progress(i, testData:size())
     	-- rgb -> yuv
     	local rgb = testData.data[i]
     	local yuv = image.rgb2yuv(rgb)
     	-- normalize y locally:
     	yuv[1] = normalization(yuv[{{1}}])
     	testData.data[i] = yuv
end
-- normalize u globally:
mean_u = provider.trainData.mean_u
std_u = provider.trainData.std_u
mean_v = provider.trainData.mean_v
std_v = provider.trainData.std_v

testData.data:select(2,2):add(-mean_u)
testData.data:select(2,2):div(std_u)
-- normalize v globally:
testData.data:select(2,3):add(-mean_v)
testData.data:select(2,3):div(std_v)
]]


-- Load the model

print "==> loading model..."
model = torch.load(model_file)

print "==> predicting..."
model:cuda()
model:evaluate()

--[[
for t = 1, testData.size() do
	xlua.progress(t, testData.size())

	local input = testData.data[t]:cuda()
	local pred = model:forward(input)
end 
]]
--preds = {}
--probs = {}
--confusion = optim.ConfusionMatrix(10)

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
		--preds[currIdx] = label[1]
        --probs[currIdx] = torch.max(outputs[k]:float())
        if torch.max(outputs[k]:float()) > 12 then
        	selectedData.data[selectedNum] = testData.data[currIdx]
 			selectedData.labels[selectedNum] = label[1]
			selectedNum = selectedNum+1
		currIdx = currIdx + 1
	end
end


--confusion:updateValids()
--print('val accuracy:', confusion.totalValid * 100)

--print(confusion)

--[[
print('==> saving predictions')
file = io.open('_predictions_augments.csv', 'w')
file:write('Id,Prediction\n')
for i, p in ipairs(preds) do
   file:write(i..','..p..'\n')
end
file:close()
]]

torch.save('selected1.t7', selectedData)
end
