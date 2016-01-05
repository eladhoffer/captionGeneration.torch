
function Key(num)
    return string.format('%06d',num)
end


return
{
    VOCAB_FILE = '/home/ehoffer/Datasets/COCO/LMDB/Vocab.t7',
    TRAINING_IMAGE_PATH = '/home/ehoffer/Datasets/COCO/train2014/', --Training images location
    TRAINING_CAPTIONS_FILE = '/home/ehoffer/Datasets/COCO/annotations/captions_train2014.json',
    VALIDATION_IMAGE_PATH = '/home/ehoffer/Datasets/COCO/val2014/',  --Validation images location
    VALIDATION_CAPTIONS_FILE = '/home/ehoffer/Datasets/COCO/annotations/captions_val2014.json',
    VALIDATION_DATA = '/home/ehoffer/Datasets/COCO/LMDB/validation/', --Validation LMDB location
    TRAINING_DATA = '/home/ehoffer/Datasets/COCO/LMDB/train/', --Training LMDB location
    ImageMinSide = 256, --Minimum side length of saved images
    Normalization = {'simple', 118.380948, 61.896913}, --Default normalization -global mean, std
    Compressed = true,
    PreTrainedCNN = '../GoogLeNet.torch/GoogLeNet_v2.t7',
    Key = Key
}
