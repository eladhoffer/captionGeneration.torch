require 'cudnn'
require 'cunn'
require 'recurrent'

function SpatialAttentivePooling(iM)
  local attention = nn.Sequential()
  attention:add(cudnn.SpatialConvolution(iM, 1, 1, 1))
  attention:add(nn.View(-1):setNumInputDims(3))
  attention:add(nn.SoftMax())
  attention:add(nn.Replicate(iM, 1, 1))
  local tbl = nn.ConcatTable():add(attention):add(nn.View(iM, -1):setNumInputDims(3))
  local m = nn.Sequential():add(tbl):add(nn.CMulTable()):add(nn.Sum(2,2))
  return m
end
function RSSA(inputSize, outputSize, iW, iH) --Recurrent Spatial Soft attention
  require 'nngraph'
      -- there will be 2 input: {input, state}
      local input = nn.Identity()()
      local state = nn.Identity()()
      local expandedVec = nn.Replicate(iW, 3, 2)(nn.Replicate(iH, 2, 1)(state))
      local x = nn.JoinTable(1,3)({input, expandedVec})
      local xConved = cudnn.SpatialConvolution(inputSize + outputSize, outputSize, 1, 1)(x)
      local pooled = SpatialAttentivePooling(outputSize)(xConved)


      local rnnModule = nn.gModule({input, state}, {pooled, nn.Identity()(pooled)})

      return nn.RecurrentContainer{
          rnnModule = rnnModule,
          initState = torch.zeros(1, outputSize),
          name = 'nn.RSSA(' .. inputSize ..'x'..iW..'x'..iH .. ' -> ' .. outputSize .. ', ' .. outputSize .. ')'
      }
  end
