require 'cunn'
require 'cudnn'
require 'nngraph'
require 'image'
require 'trepl'
require 'recurrent'

cutorch.setDevice(2)
local config = require 'Config'
config.InputSize = {3,224,224}
config.SentenceLength = 10
local numImages = 20
local numWords = 15
local data =require 'Data'
local imgs = torch.ByteTensor(numImages, unpack(config.InputSize))
local captions
imgs, captions = data.ValDB:cacheSeq(Key(200),numImages, imgs)
imgs = imgs:cuda()
imgs:add(-config.Normalization[2]):div(config.Normalization[3])
local cnnModel = torch.load(config.PreTrainedCNN):cuda()
local removeAfter = config.FeatLayerCNN
for i = #cnnModel, removeAfter ,-1 do
    cnnModel:remove(i)
end
local imgNum = 1
local modelConfig = torch.load('./Results/captionInceptionv3/Net_20.t7')
local textEmbedder = modelConfig.textEmbedder:cuda()
local imageEmbedder = modelConfig.imageEmbedder:cuda()
local classifier = modelConfig.classifier:cuda()
local recurrent = modelConfig.recurrent:cuda()
local model = nn.Sequential():add(recurrent):add(classifier)
local inputSize = modelConfig.inputSize
local vocab = config.Vocab
local decoder = data.decoder

model:evaluate()
imageEmbedder:evaluate()
textEmbedder:evaluate()
cnnModel:evaluate()
local temperature = 0--temperature or 1
local function beamSearch(model, firstToken, numDraw, numKeep, temp)
  local temperature = temp or 1
  local numKeep = numKeep or 20
  local numDraw = numDraw or 5
  local preds = model:forward(firstToken)

  preds:div(temperature) -- scale by temperature
  local probs = torch.exp(preds)
  probs:cdiv(probs:sum(1):expandAs(probs)) -- renormalize so probs sum to one

  local draws, drawsIdx = probs:sort(2)
  draws:cmul(currProbs:expandAs(draws))
  for i = 1, draws:size() do
    return num
  end
end

recurrent:evaluate()
recurrent:single()
local numVecs = torch.IntTensor(numImages, numWords):zero()
local imageRep = cnnModel:forward(imgs)
local embeddedImg = imageEmbedder:forward(imageRep):squeeze()
model:zeroState()
local pred = model:forward(embeddedImg)
local _, wordNums = pred:max(2)
for j=1, numWords do
  numVecs:select(2,j):copy(wordNums)
  embedded = textEmbedder:forward(wordNums):squeeze()
  pred = model:forward(embedded)
  _, wordNums = pred:max(2)
end
for i=1, numImages do
  print(decode(numVecs[i], vocab['.']))
end
require 'image'
image.display(imgs:float())
