require 'xlua'
require 'lmdb'


local DataProvider = require 'DataProvider'
local config = require 'Config'
config.Vocab = torch.load(config.VOCAB_FILE)

local numMinOccur = 10
for word, count in pairs(config.Vocab.WordsCount) do
  if count < numMinOccur then
    config.Vocab.Vocab[word] = nil
  end
end
config.Vocab = config.Vocab.Vocab

local newCount = 1
for word, _ in pairs(config.Vocab) do
  config.Vocab[word] = newCount
  newCount = newCount + 1
end
--require 'tds'
--
--config.VocabVec = torch.load('../../../Datasets/GloVe/vectors.6B.100d.txt_output.t7')
--config.Vocab = {}
--local Vecs = torch.FloatTensor(#config.VocabVec, 100)
--local num = 1
--for word, vec in pairs(config.VocabVec) do
--  config.Vocab[word] = num
--  Vecs[num]:copy(vec)
--  num = num+1
--end
--config.VocabVec = Vecs
function ExtractFromLMDB(data)
    require 'image'

    function encode(str)
      local vec = torch.IntTensor(config.SentenceLength):fill(config.Vocab['.'])
      local words = str:split(' ')
      for i=1,math.min(config.SentenceLength,#words) do
        vec[i] = config.Vocab[words[i]] or config.Vocab['<UNK>']
      end
      return vec
    end

    -- local reSample = function(sampledImg)
    --     local sizeImg = sampledImg:size()
    --     local szx = torch.random(math.ceil(sizeImg[3]/4))
    --     local szy = torch.random(math.ceil(sizeImg[2]/4))
    --     local startx = torch.random(szx)
    --     local starty = torch.random(szy)
    --     return image.scale(sampledImg:narrow(2,starty,sizeImg[2]-szy):narrow(3,startx,sizeImg[3]-szx),sizeImg[3],sizeImg[2])
    -- end
    -- local rotate = function(angleRange)
    --     local applyRot = function(Data)
    --         local angle = torch.randn(1)[1]*angleRange
    --         local rot = image.rotate(Data,math.rad(angle),'bilinear')
    --         return rot
    --     end
    --     return applyRot
    -- end

    local img = data.Data
    if config.Compressed then
        img = image.decompressJPG(img,3,'byte')
    end
    if math.min(img:size(2), img:size(3)) ~= config.ImageMinSide then
         img = image.scale(img, '^' .. config.ImageMinSide)
    end
    --
    -- if config.Augment == 3 then
    --     img = rotate(0.1)(img)
    --     img = reSample(img)
    -- elseif config.Augment == 2 then
    --     img = reSample(img)
    -- end
    --local startX = math.ceil((img:size(3)-config.InputSize[3]+1)/2)
    --local startY = math.ceil((img:size(2)-config.InputSize[2]+1)/2)
    -- if rand then
      local startX = math.random(img:size(3)-config.InputSize[3]+1)
      local startY = math.random(img:size(2)-config.InputSize[2]+1)


    img = img:narrow(3,startX,config.InputSize[3]):narrow(2,startY,config.InputSize[2])
    local hflip = torch.random(2)==1
     if hflip then
         img = image.hflip(img)
     end
    local randOrder = torch.randperm(#data.Caption)
    local caption = data.Caption[randOrder[1]]
    for i=2, randOrder:size(1) do
      caption = caption .. ' ' .. data.Caption[randOrder[i]]
    end
    return img, encode(caption)
end



function Keys(tensor)
    local tbl = {}
    for i=1,tensor:size(1) do
        tbl[i] = config.Key(tensor[i])
    end
    return tbl
end



local TrainDB = DataProvider.LMDBProvider{
    Source = lmdb.env({Path = config.TRAINING_DATA, RDONLY = true}),
    ExtractFunction = ExtractFromLMDB
}
local ValDB = DataProvider.LMDBProvider{
    Source = lmdb.env({Path = config.VALIDATION_DATA , RDONLY = true}),
    ExtractFunction = ExtractFromLMDB
}

local decoder = {}
for word,num in pairs(config.Vocab) do
  decoder[num] = word
end

function decode(vec)
  local str = decoder[vec[1]]
  for i=2, vec:nElement() do
    str = str .. ' ' .. decoder[vec[i]]
  end
  return str
end

return {
    ValDB = ValDB,
    TrainDB = TrainDB,
    decoder = decoder,
    decode = decode
}
