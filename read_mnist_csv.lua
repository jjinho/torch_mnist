-- Code modified from chetannaik@GitHub

-----------------------------------------------------------------------
-- Split string
-- Source code from lua-user.org/wiki/Split_Join
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end


-----------------------------------------------------------------------
-- Paths to files
local trainFilePath = 'train.csv'
local testFilePath = 'test.csv'

-----------------------------------------------------------------------
-- Training Data
-- Count number of rows and columns in file
local i = 0
for line in io.lines(trainFilePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

local ROWS = i - 1  -- Minus 1 because of header

-- Read data from CSV to tensor
local csvFile = io.open(trainFilePath, 'r')
local header = csvFile:read()

-- Label tensor
-- Will hold the first column from all lines
local trainLabel = torch.DoubleTensor(ROWS)

-- Data tensor
-- ROWS x 28 x 28
local trainData = torch.DoubleTensor(ROWS, 1, 28, 28)

-- Training data
local i = 1
for line in csvFile:lines('*l') do
  print(i)
  local l = line:split(',')
  for j, val in ipairs(l) do
    if j == 1 then
      if val == '0' then
        val = 10
      end
      trainLabel[i] = val
    else
      -- j ranges from 0 .. 783
      j = j - 2

      -- x and y are indices (that start at 1) calculated from j
      local x = j % 28 + 1
      local y = math.floor(j / 28) + 1
      -- print(i)
      -- print(x)
      -- print(y)
      -- print(j)
      trainData[i][1][x][y] = val
    end
  end
  i = i + 1
end

csvFile:close()

-----------------------------------------------------------------------
-- Test Data
-- Count number of rows and columns in file
ROWS = 0
COLS = 0

local i = 0
for line in io.lines(testFilePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

local ROWS = i - 1  -- Minus 1 because of header

-- Read data from CSV to tensor
csvFile = io.open(testFilePath, 'r')
header = csvFile:read()

-- Data tensor
-- ROWS x 28 x 28
local testData = torch.DoubleTensor(ROWS, 1, 28, 28)

-- Training data
local i = 1
for line in csvFile:lines('*l') do
  print(i)
  local l = line:split(',')
  for j, val in ipairs(l) do
    -- j ranges from 0 .. 783
    j = j - 1

    -- x and y are indices (that start at 1) calculated from j
    local x = j % 28 + 1
    local y = math.floor(j / 28) + 1
    -- print(i)
    -- print(x)
    -- print(y)
    -- print(j)
    testData[i][1][x][y] = val
  end
  i = i + 1
end

csvFile:close()


-- Serialize tensor
local outputTrainFilePath = 'train_mnist.th7'
local outputTestFilePath = 'test_mnist.th7'
torch.save(outputTrainFilePath, trainData)
torch.save(outputTestFilePath, testData)

local outputTrainFilePath2 = 'train_label_mnist.th7'
torch.save(outputTrainFilePath2, trainLabel)

-- Deserialize tensor object
local restored_data = torch.load(outputTrainFilePath)
local restored_data2 = torch.load(outputTrainFilePath2)
local restored_data3 = torch.load(outputTestFilePath)

-- Make test
print(trainData:size())
print(restored_data:size())

print(trainLabel:size())
print(restored_data2:size())

print(testData:size())
print(restored_data3:size())

