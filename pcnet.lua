require 'cunn'
require 'nn'

-----------------------------------------------------------------------
-- Data
trainset = torch.load('train_mnist.th7')
trainsetLabels = torch.load('train_label_mnist.th7')

trainData = trainset[{ {1,40000}, {}, {}, {} }]
trainLabel = trainsetLabels[{ {1,40000} }]

validData = trainset[{ {40001, 42000}, {}, {}, {} }]
validLabel = trainsetLabels[{ {40001, 42000} }]

-- Preparing training data for use with nn.StochasticGradient
train = {
	data = trainData,
	label = trainLabel
}

-- nn.StochasticGradient requires that the training set have an index
setmetatable(train, 
	{__index = function(t, i) return { t.data[i], t.label[i] } end} );

-- nn.StochasticGradient requires that the training set return size
function train:size() return self.data:size(1) end

-- Prepare validation data for use with nn.StochasticGradient
validate = {
	data = validData,
	label = validLabel
}

setmetatable(validate, 
	{__index = function(t, i) return { t.data[i], t.label[i]} end} );

function validate:size() return self.data:size(1) end

-----------------------------------------------------------------------
-- Classes
-- The Training data substitutes 10 for 0 since this breaks
-- nn.ClassNLLCriterion if we leave it as 0
classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}

-----------------------------------------------------------------------
-- Preprocessing training data
mean = {}
stdv = {}

for i=1,40000 do
	mean[i] = train.data[{ {i}, {}, {}, {} }]:mean()
	train.data[{ {i}, {}, {}, {} }]:add(-mean[i])
	
	stdv[i] = train.data[{ {i}, {}, {}, {} }]:std()
	train.data[{ {i}, {}, {}, {} }]:div(stdv[i])
end

-- Preprocessing validation data
for i=1,2000 do
	mean[i] = validate.data[{ {i}, {}, {}, {} }]:mean()
	validate.data[{ {i}, {}, {}, {} }]:add(-mean[i])
	
	stdv[i] = validate.data[{ {i}, {}, {}, {} }]:std()
	validate.data[{ {i}, {}, {}, {} }]:div(stdv[i])
end

-----------------------------------------------------------------------
-- Convolution Neural Network
-- Modeled after LeNet5
net = nn.Sequential()
--net:add(nn.Reshape(1, 28, 28))
-- 1 x 28 x 28 -> 6 x 24 x 24
net:add(nn.SpatialConvolution(1, 6, 5, 5))
net:add(nn.ReLU())
-- 6 x 24 x 24 -> 6 x 12 x 12
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.ReLU())
-- 6 x 12 x 12 -> 16 x 8 x 8
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())
-- 16 x 8 x 8 -> 16 x 4 x 4
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.View(16 * 4 * 4))
net:add(nn.Linear(16 * 4 * 4, 120))
net:add(nn.ReLU())
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())
net:add(nn.Linear(84, #classes))
net:add(nn.LogSoftMax())

-- Print the CNN
print(net)

-----------------------------------------------------------------------
-- Criterion
criterion = nn.ClassNLLCriterion()

-- CUDA
net = net:cuda()
criterion = criterion:cuda()
train.data = train.data:cuda()

-----------------------------------------------------------------------
-- Training
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 20

trainer:train(train)

-----------------------------------------------------------------------
-- Validation

-- CUDA
validate.data = validate.data:cuda()
correct = 0
for i=1,2000 do
	local groundtruth = validate.label[i]
	local prediction = net:forward(validate.data[i])
	
	-- true here means sorting in descending order
	local confidences, indices = torch.sort(prediction, true)
	
	if groundtruth == indices[1] then
		correct = correct + 1
	end
end

print(correct)

-----------------------------------------------------------------------
-- Save CNN
torch.save('pcnet.cnn', net)

-----------------------------------------------------------------------
-- Load the Test data
testset = torch.load('test_mnist.th7')

-----------------------------------------------------------------------
-- Preprocessing the Test data
test = {
  data = testset
}

setmetatable(test,
  {__index = function(t, i) return { t.data[i] } end} );

function test:size() return self.data:size(1) end

-- Preprocessing
local testMean = {}
local testStdv = {}

for i=1,28000 do
  testMean[i] = test.data[{ {i}, {}, {}, {} }]:mean()
  test.data[{ {i}, {}, {}, {} }]:add(-testMean[i])

  testStdv[i] = test.data[{ {i}, {}, {}, {} }]:std()
  test.data[{ {i}, {}, {}, {} }]:div(testStdv[i])
end

-----------------------------------------------------------------------
-- CUDA
test.data = test.data:cuda()

-----------------------------------------------------------------------
-- Get Predictions
local testPredictions = {}

for i=1,28000 do
  local testPrediction = net:forward(test.data[i])
  local testConfidences, testIndices = torch.sort(testPrediction, true)
  testPredictions[i] = testIndices[1]
end

print(testPredictions)

-----------------------------------------------------------------------
-- Print Predictions
-- Making an output file compatible with Kaggle's submission criteria
file = io.open('mnist_predictions.csv', 'w')
file:write('ImageId,Label\n')
for i=1,28000 do
  file:write(i)
  file:write(',')
  -- Have to convert 10 back to 0
  if testPredictions[i] == 10 then
    file:write(0)
  else
    file:write(testPredictions[i])
  end
  file:write("\n")
end
file:close()

