require 'xlua'
require 'image'
require 'torch'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
]]

print(opt)

do -- data augmentation module
  
  -- horizontal flip
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end

  -- rotate
  local BatchRotate, parent = torch.class('nn.BatchRotate', 'nn.Module')

  function BatchRotate:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchRotate:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local rotate_mask = torch.randperm(bs):le(bs/2)
      for i=1, input:size(1) do
        if rotate_mask[i] == 1 then
          local theta = torch.uniform(-15,15)
          local rad = math.rad(theta)
          input[i] = image.rotate(input[i], rad) 
        end
      end
    end
    self.output:set(input)
    return self.output
  end

  -- scale + crop
  local BatchScale, parent = torch.class('nn.BatchScale', 'nn.Module')
    
  function BatchScale:__init()
    parent.__init(self)
    self.train = true
  end
    
  function BatchScale:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local scale_mask = torch.randperm(bs):le(bs/2)
      for i=1, input:size(1) do
        if scale_mask[i] == 1 then 
          local ratio = torch.uniform(1,1.4)
          local width = input[i]:size(2)*ratio
          local height = input[i]:size(3)*ratio
          local temp = image.scale(input[i], width, height)
          local left = width/2 - 48
          local right = left + 96
          local bottom = height/2 - 48
          local top = bottom + 96
          image.crop(input[i], temp, left, bottom, right, top)
        end
      end
    end
    self.output:set(input)
    return self.output
  end

  -- color augmentation
  local BatchColor, parent = torch.class('nn.BatchColor', 'nn.Module')
    
  function BatchColor:__init()
    parent.__init(self)
    self.train = true
  end
    
  function BatchColor:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local color_mask = torch.randperm(bs):le(bs/2)
      for i=1, input:size(1) do
        if color_mask[i] == 1 then
          image.rgb2hsv(input[i], input[i])
          local a = torch.uniform(-0.1,0.1)
          input[i] = input[i] + a
          image.hsv2rgb(input[i], input[i])
        end
      end
    end
    self.output:set(input)
    return self.output
  end
end


print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchColor():float())
model:add(nn.BatchFlip():float())
model:add(nn.BatchRotate():float())
model:add(nn.BatchScale():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/'..opt.model..'.lua'):cuda())
model:get(5).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(6), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = Provider()
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function val()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,provider.valData.data:size(1),bs do
    local outputs = model:forward(provider.valData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.valData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)
  
  if valLogger then
    paths.mkdir(opt.save)
    valLogger:add{train_acc, confusion.totalValid * 100}
    valLogger:style{'-','-'}
    valLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/val.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(6))
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  val()
end


