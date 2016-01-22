
function Key(num)
  return string.format('%06d',num)
end


local config = {model = 'inception_v2'} --default

if config.model == 'inception_v2' then
  config.PreTrainedCNN = '../GoogLeNet.torch/GoogLeNet_v2.t7' --'../inception-v3.torch/inception_v3.net',--
  config.Normalization = {'simple', 118.380948, 61.896913} --Default normalization -global mean, std
  config.FeatLayerCNN = 27
  config.NumFeatsCNN = 1024
elseif config.model == 'inception_v3' then
  config.PreTrainedCNN = '../inception-v3.torch/inception_v3.net'--
  config.Normalization = {'simple', 128, 128}--Inception-v3 case
  config.FeatLayerCNN = 28
  config.NumFeatsCNN = 2048
end

config.VOCAB_FILE = '/home/ehoffer/Datasets/COCO/LMDB/Vocab.t7'
config.TRAINING_IMAGE_PATH = '/home/ehoffer/Datasets/COCO/train2014/' --Training images location
config.TRAINING_CAPTIONS_FILE = '/home/ehoffer/Datasets/COCO/annotations/captions_train2014.json'
config.VALIDATION_IMAGE_PATH = '/home/ehoffer/Datasets/COCO/val2014/'  --Validation images location
config.VALIDATION_CAPTIONS_FILE = '/home/ehoffer/Datasets/COCO/annotations/captions_val2014.json'
config.VALIDATION_DATA = '/home/ehoffer/Datasets/COCO/LMDB/validation/' --Validation LMDB location
config.TRAINING_DATA = '/home/ehoffer/Datasets/COCO/LMDB/train/' --Training LMDB location
config.InputSize = {3,224,224}
config.ImageMinSide = 256 --Minimum side length of saved images
config.Compressed = true
config.Key = Key
return config
