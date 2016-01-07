function SpatialAttentivePooling(iM, iW, iH)
  local attention = nn.Sequential()
  attention:add(cudnn.SpatialConvolution(iM, 1, 1, 1))
  attention:add(nn.View(-1):setNumInputDim(3))
  attention:add(nn.SoftMax())
  attention:add(nn.Repicate(iM, 1, 1))
  local tbl = nn.ConcatTable():add(attention):add(nn.View(iM, -1):setNumInputDim(3))
  local m = nn.Sequential():add(tbl):add(nn.CMulTable()):add(nn.Sum(2,2))
return m
