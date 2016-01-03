require 'image'
require 'xlua'
require 'lmdb'
require 'pl.stringx'

local gm = require 'graphicsmagick'
local DataProvider = require 'DataProvider'
local config = require 'Config'


local json = require 'dkjson'
local removedSymbols = '[\n,"?.!/\\<>()]'
local vocab = {['.'] = 1, ['<UNK>'] = 2}
local trainingCaptions = {}
local wordsCount = {}
local validationCaptions = {}
local currentNum = 1
local longestCaption = 1
--count num words (in case of existing vocab)
for _ in pairs(vocab) do currentNum = currentNum + 1 end


local function parse(str, addWords)
  local str = str:gsub(removedSymbols, ' '):gsub('%s+',' '):gsub('%s$',''):lower()
  str = str .. ' .' --ensure ends with dot
  local words = str:split(' ')
  longestCaption = math.max(longestCaption, #words)
  for i=1, #words do
    local currWord = words[i]
    if not vocab[currWord] and addWords then
      vocab[currWord] = currentNum

      currentNum = currentNum + 1
    end
    wordsCount[currWord] = wordsCount[currWord] or 0
    wordsCount[currWord] = wordsCount[currWord] + 1
  end
  return str
end

local decodedJson = json.decode(io.open(config.TRAINING_CAPTIONS_FILE):read('*all')).annotations
for _, content in pairs(decodedJson) do
  trainingCaptions[content.image_id] = trainingCaptions[content.image_id] or {}
  table.insert(trainingCaptions[content.image_id], parse(content.caption, true))
end

decodedJson = json.decode(io.open(config.VALIDATION_CAPTIONS_FILE):read('*all')).annotations
for _, content in pairs(decodedJson) do
  validationCaptions[content.image_id] = validationCaptions[content.image_id] or {}
  table.insert(validationCaptions[content.image_id], parse(content.caption, false))
end

-- -------------------------------Settings----------------------------------------------
--
local PreProcess = function(Img)
  local im = image.scale(Img, '^' .. config.ImageMinSide) --minimum side of ImageMinSide

  if im:dim() == 2 then
    im = im:reshape(1,im:size(1),im:size(2))
  end
  if im:size(1) == 1 then
    im=torch.repeatTensor(im,3,1,1)
  end
  if im:size(1) > 3 then
    im = im[{{1,3},{},{}}]
  end
  return im
end
--
--
local LoadImgData = function(filename)
  local img = gm.Image(filename):toTensor('byte','RGB','DHW')
  if img == nil then
    print('Image is buggy')
    print(filename)
    os.exit()
  end
  img = PreProcess(img)
  if config.Compressed then
    return image.compressJPG(img)
  else
    return img
  end
end
--
function ImageNum(filename)
  local name = paths.basename(filename,'.jpg')
  local substring = string.split(name,'_')
  local num = tonumber(substring[#substring])
  return num

end
--
function readImgFile(filename)
  local file = torch.DiskFile(filename, 'r')
  file:seekEnd()
  local length = file:position() - 1
  file:seek(1)
  local byteVec = torch.ByteTensor(length)
  file:readByte(byteVec:storage())
  file:close()
  return byteVec
end
function LMDBFromFilenames(filenamesProvider,env, captions)
  env:open()
  local txn = env:txn()
  local cursor = txn:cursor()
  for i=1, filenamesProvider:size() do
    local filename = filenamesProvider:getItem(i)
    local imgNum = ImageNum(filename)
    local data = {Data = LoadImgData(filename), Num = imgNum, Caption = captions[imgNum]}

    cursor:put(config.Key(i),data, lmdb.C.MDB_NODUPDATA)
    if i % 1000 == 0 then
      txn:commit()
      print(env:stat())
      collectgarbage()
      txn = env:txn()
      cursor = txn:cursor()
    end
    xlua.progress(i,filenamesProvider:size())
  end
  txn:commit()
  env:close()

end


TrainingFiles = DataProvider.FileSearcher{
  Name = 'TrainingFilenames',
  CachePrefix = config.TRAINING_DATA,
  MaxNumItems = 1e8,
  CacheFiles = true,
  PathList = {config.TRAINING_IMAGE_PATH},
  Verbose = true
}

local ValidationFiles = DataProvider.FileSearcher{
    Name = 'ValidationFilenames',
    CachePrefix = config.VALIDATION_DATA,
    MaxNumItems = 1e8,
    PathList = {config.VALIDATION_IMAGE_PATH},
    Verbose = true
}

local TrainDB = lmdb.env{
  Path = config.TRAINING_DATA,
  Name = 'TrainDB'
}

local ValDB = lmdb.env{
    Path = config.VALIDATION_DATA,
    Name = 'ValDB'
}

TrainingFiles:shuffleItems()
LMDBFromFilenames(ValidationFiles, ValDB, validationCaptions)
LMDBFromFilenames(TrainingFiles, TrainDB, trainingCaptions)
torch.save(config.VOCAB_FILE, {Vocab = vocab, WordsCount = wordsCount})
print(vocab)
print("Num Words: " .. currentNum-1)
print("Longest Caption: " .. longestCaption, " Words")
