-- kmeans script

require 'xlua'
require 'image'
require 'torch'
require 'optim'
require 'nn'
require 'unsup'

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


function patchify(data, n_patches, kW, kH, dW, dH, seed)
  -- extract patches from images
  -- basic var
  seed = seed or 86
  dW = dW or 1
  dH = dH or 1
  local h = data.X:size()[3]
  local w = data.X:size()[4]
  local n = data.size
  local channel = data.X:size()[2]
  patches = torch.Tensor(n*n_patches, channel*kW*kH)

  -- calculate total number of patches 
  local n_row = (h - kH) / dH + 1
  local n_col = (w - kW) / dW + 1
  local all_idx = torch.range(1, n_row * n_col)

  -- extract n_patches randomly from each image   
  torch.manualSeed(seed)
  for i = 1, n do   -- for each image
    xlua.progress(i, n)   -- display progress
    local first = (i-1)*n_patches+1   -- index of first patch for this image
    --print(first)
    local idx = torch.multinomial(all_idx, n_patches)   -- randomly choose n_patches
    for j = 1, n_patches do
        local k = idx[j]
        --print(k)
        -- find position of the patch
        local x = ((k-1) % n_col) * dW + 1
        --print(x)
        local y = torch.floor((k-1) / n_col) * dH + 1
        --print(y)
        -- copy patch to new tensor
        patches[{{first+j-1}, {}}] = torch.Tensor(channel*kW*kH):copy(data.X[{{i}, {}, {x, x+kW-1}, {y, y+kH-1}}])
    end
  end
  return patches
end

print ("==> load data...")
extra = torch.load('stl-10/extra.t7b')
X = torch.ByteTensor(#extra.data[1], 3, 96, 96)

for i = 1, #extra.data[1] do
    X[i] = extra.data[1][i]
end

X = X:float()

--normalize

print ("==> normalize data...")
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,X:size(1) do
     xlua.progress(i, X:size(1))
     -- rgb -> yuv
     local rgb = X[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     X[i] = yuv
end
  -- normalize u globally:
local mean_u = X:select(2,2):mean()
local std_u = X:select(2,2):std()
X:select(2,2):add(-mean_u)
X:select(2,2):div(std_u)
-- normalize v globally:
local mean_v = X:select(2,3):mean()
local std_v = X:select(2,3):std()
X:select(2,3):add(-mean_v)
X:select(2,3):div(std_v)

print ("==> data augmentation...")
--store data to augmented
n = X:size(1)*2
augmented = {
    X = torch.Tensor(n, 3, 96, 96),
    size = n
}
for i = 1, X:size(1) do
    augmented.X[i] = X[i]
end

--data augmentation
model = nn.Sequential()
--model:add(nn.BatchColor():float())
model:add(nn.BatchFlip():float())
model:add(nn.BatchRotate():float())
model:add(nn.BatchScale():float())

outputs = model:forward(X)
for i = 1, X:size(1) do
    augmented.X[i+X:size(1)] = outputs[i]
end


filtersize = 7
n_patches = 10
n_channel = 3

print ("==> get patches...")
pats = patchify(augmented, n_patches, filtersize, filtersize)
print ("==> whiten data...")
whitened_pats = unsup.zca_whiten(pats,nil,nil,nil,epsilon)
num_kernels = 1024
niters = 500
print ("kmeans starts...")
cents = unsup.kmeans(whitened_pats, num_kernels, niters, verbose==true):reshape(num_kernels, n_channel, filtersize, filtersize)

file = 'kmeans_'..num_kernels..'.t7'
print('==> saving centroids to disk: ' .. file)
torch.save(file, cents)
