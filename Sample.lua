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
local removeAfter = 26
for i=30, removeAfter ,-1 do
  cnnModel:remove(i)
end
local imgNum = 1
local modelConfig = torch.load('./Results/captionGRU_512/Net_7.t7')
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
local function smpWord(preds)

  if temperature == 0 then
    local _, num = preds:max(2)
    return num
  else
    preds:div(temperature) -- scale by temperature
    local probs = torch.exp(preds):squeeze()
    probs:div(probs:sum()) -- renormalize so probs sum to one
    local num = torch.multinomial(probs:float(), 1):view(1,-1)
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
