--[[

This is a Torch implementation of InfoGAN.

"InfoGAN: Interpretable Representation Learning by Information Maximizing
 Generative Adversarial Nets"
  - Chen et al, http://arxiv.org/abs/1606.03657

--]]

require('torch')    -- Essential Torch utilities
require('image')    -- Torch image handling
require('nn')       -- Neural network building blocks
require('optim')    -- Optimisation algorithms
require('cutorch')  -- 'torch' on the GPU
require('cunn')     -- 'nn' on the GPU

local tnt = require('torchnet')
local pl = require('pl.import_into')()

package.path = package.path .. ';./src/?.lua;./src/?/init.lua'

util = paths.dofile('../util/util.lua')
local model_utils = require 'util.model_utils'
local pdist = require('pdist')
local MnistDataset = require('MnistDataset')
local model_builder = require('dcgan_model_builder')
unpack=unpack or table.unpack
--- OPTIONS ---

local opts = pl.lapp [[
Trains an InfoGAN network
  --epochs (default 100) Number of training epochs
  --updates-per-epoch (default 1000) Number of batches per epoch
  --batch-size (default 128) Number of examples per batch
  --disc-learning-rate (default 2e-4) Discriminator network learning rate
  --gen-learning-rate (default 2e-4) Generator network learning rate
  --info-reg-coeff (default 1.0) "lambda" from the InfoGAN paper
  --rng-seed (default 1234) Seed for random number generation
  --gen-inputs (default 100) Number of inputs to the generator network
  --uniform-salient-vars (default 2) Number of non-categorical salient inputs
  --ngen (default 3) Number of generators to use
  --ndf (default 64) Number of discriminator filters
  --ngf (default 64) Number of generator filters
  --batchSize (default 64) Number of images to process together
  --loadSize (default 96) The load size
  --fineSize (default 64) The final crop size
  --nThreads (default 4) The number of dataloading threads to use
  --dataset (default 'folder') The type of dataset to use
  --DATA_ROOT (default 'celebA') The dataset to be used
  --exp-name (default 'dcgan') The experiment name
  --n-sets-categorical (default 1) The sets of categorical variables
  --gpu (default 1) Which GPU to use
  --continue-train (default true)
  --start-epoch (default 1)
  --lambda-compete (default 0.01)
  --out-size (default 64)
]]
print(opts)
cutorch.setDevice(opts.gpu)
local n_epochs = opts.epochs
local n_updates_per_epoch = opts.updates_per_epoch
local batch_size = opts.batch_size
local info_regularisation_coefficient = opts.info_reg_coeff
local disc_learning_rate = opts.disc_learning_rate
local gen_learning_rate = opts.gen_learning_rate
local rng_seed = opts.rng_seed
local n_gen_inputs = opts.gen_inputs
local n_sets_categorical=opts.n_sets_categorical
local n_salient_vars = 10*n_sets_categorical + opts.uniform_salient_vars
local ngen = opts.ngen
local n_noise_vars = n_gen_inputs - n_salient_vars
local exp_name=opts.exp_name
local lambda_compete = opts.lambda_compete
opts.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opts.manualSeed)
torch.manualSeed(opts.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('../dataset/data.lua')
local data = DataLoader.new(opts.nThreads, opts.dataset, opts)
print("Dataset: " .. opts.dataset, " Size: ", data:size())

assert(n_salient_vars >= 10 and n_salient_vars < n_gen_inputs,
  'At least one generator input must be non-salient noise')

--- INIT ---
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

-- Set manual seeds for reproducible RNG
torch.manualSeed(rng_seed)
cutorch.manualSeedAll(rng_seed)
math.randomseed(rng_seed)

torch.setdefaulttensortype('torch.FloatTensor')

--- DATA ---

--local train_data = MnistDataset.new('data/mnist/train_32x32.t7')
--local train_iter = train_data:make_iterator(batch_size)

--- MODEL ---

local dist = pdist.Hybrid()
for i=1,n_sets_categorical do
  dist:add(pdist.Categorical{n = 10, probs = torch.CudaTensor(10):fill(1 / 10)})
--  :add(pdist.Categorical{n = 10, probs = torch.CudaTensor(10):fill(1 / 10)})
end  
if opts.uniform_salient_vars ~=0 then
  dist:add(pdist.Gaussian{
    n = n_salient_vars - 10*n_sets_categorical,
    mean = torch.CudaTensor(n_salient_vars - 10*n_sets_categorical):fill(0),
    stddev = torch.CudaTensor(n_salient_vars - 10*n_sets_categorical):fill(1),
    fixed_stddev = true
  })
end
local discriminator_body=nil
local discriminator_head=nil
local info_head=nil
G={}
G['generator1'],discriminator_body,discriminator_head,info_head =
  model_builder.build_infogan(n_gen_inputs, dist:n_params(),opts)
G.relu=nn.ReLU()
G.cosine=nn.CosineDistance()
local discriminator = nn.Sequential()
  :add(discriminator_body)
  :add(nn.ConcatTable()
    :add(discriminator_head)
    :add(info_head)
  )
discriminator:apply(weights_init)
for i=2,ngen do
    G['generator'..i]=G.generator1:clone()
    G['generator'..i]:apply(weights_init)
end

--generator:cuda()
for k,net in pairs(G) do net:cuda() end
discriminator:cuda()

local model_dir = pl.path.join('out',exp_name, 'models')
pl.dir.makepath(model_dir)

local model_disc_file = pl.path.join(model_dir, 'infogan_disc.t7')

local model_gen_file = pl.path.join(model_dir, 'infogan_gen.t7')

if opts.continue_train==1 then
    discriminator=torch.load(model_disc_file)
    G=torch.load(model_gen_file).G
end
--- CRITERIA ---

local disc_head_criterion = nn.BCECriterion()
--local disc_head_criterion = nn.CrossEntropyCriterion()
local info_head_criterion = pdist.MutualInformationCriterion(dist)
local compete_criterion= nn.AbsCriterion()
disc_head_criterion:cuda()
info_head_criterion:cuda()
compete_criterion:cuda()

-- LOGGING --

local log_text = require('torchnet.log.view.text')

local log_keys = {'epoch', 'fake_loss', 'info_loss',
  'real_loss', 'gen_loss', 'time'}

local log = tnt.Log{
  keys = log_keys,
  onFlush = {
    log_text{
      keys = log_keys,
      format = {'epoch=%3d', 'fake_loss=%8.6f', 'info_loss=%8.6f',
        'real_loss=%8.6f', 'gen_loss=%8.6f', 'time=%5.2fs'}
    }
  }
}

--- TRAIN ---
local gen_inputs=torch.CudaTensor(ngen,opts.batchSize,n_gen_inputs)
local real_input = torch.CudaTensor()
local gen_input = torch.CudaTensor()
local fake_input = torch.CudaTensor()
local disc_target = torch.CudaTensor()
local score_D_cache=torch.Tensor(ngen,opts.batchSize):cuda()
local feature_cache=torch.Tensor(ngen,opts.batchSize,512*4*4 ):cuda()
local sum_score_D=torch.Tensor(opts.batchSize):cuda()
-- Flatten network parameters
local disc_params, disc_grad_params = discriminator:getParameters()
local gen_params, gen_grad_params = model_utils.combine_all_parameters(G)

-- Meters for gathering statistics to log
local fake_loss_meter = tnt.AverageValueMeter()
local info_loss_meter = tnt.AverageValueMeter()
local real_loss_meter = tnt.AverageValueMeter()
local gen_loss_meter = tnt.AverageValueMeter()
local time_meter = tnt.TimeMeter()

local real_label=1
local fake_label = 0
-- Calculate outputs and gradients for the discriminator
local do_discriminator_step = function(new_params)
  if new_params ~= disc_params then
    disc_params:copy(new_params)
  end

  disc_grad_params:zero()

  local batch_size = real_input:size(1)
  disc_target:resize(batch_size, 1)
  gen_input:resize(batch_size, n_gen_inputs)

  local loss_real = 0
  local loss_fake = 0
  local loss_info = 0

  -- Train with real images (from dataset)
  local dbodyout = discriminator_body:forward(real_input)

  local dheadout = discriminator_head:forward(dbodyout)
  disc_target:fill(real_label)
  loss_real = disc_head_criterion:forward(dheadout, disc_target)
  local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
  local dloss_ddheadin = discriminator_head:backward(dbodyout, dloss_ddheadout)
  discriminator_body:backward(real_input, dloss_ddheadin)
  sum_score_D:zero()
  -- Train with fake images (from generator)
  for i=1,ngen do
     dist:sample(gen_input:narrow(2, 1, n_salient_vars), dist.prior_params)
     gen_input:narrow(2, n_salient_vars + 1, n_noise_vars):normal(0, 1)
     G['generator'..i]:forward(gen_input)
     gen_inputs[i]=gen_input
     fake_input:resizeAs(G['generator'..i].output):copy(G['generator'..i].output)
     local dbodyout = discriminator_body:forward(fake_input)
     local dheadout = discriminator_head:forward(dbodyout)
     score_D_cache[i]=dheadout
     sum_score_D=sum_score_D + score_D_cache[i]
     feature_cache[i]=dbodyout:reshape(opts.batchSize,512*4*4)
     disc_target:fill(fake_label)
     loss_fake =loss_fake+ disc_head_criterion:forward(dheadout, disc_target)
     local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
     local dloss_ddheadin = discriminator_head:backward(dbodyout, dloss_ddheadout)
     discriminator_body:backward(fake_input, dloss_ddheadin)

     local iheadout = info_head:forward(dbodyout)
     local info_target = gen_input:narrow(2, 1, n_salient_vars)
     loss_info = loss_info+info_head_criterion:forward(iheadout, info_target) * info_regularisation_coefficient
     assert(loss_info == loss_info, 'info loss is nan')
     local dloss_diheadout = info_head_criterion:backward(iheadout, info_target)
     dloss_diheadout:mul(info_regularisation_coefficient)
     local dloss_diheadin = info_head:backward(dbodyout, dloss_diheadout)
     discriminator_body:backward(fake_input, dloss_diheadin)
  end
  -- Update average value meters
  real_loss_meter:add(loss_real)
  fake_loss_meter:add(loss_fake)
  info_loss_meter:add(loss_info)

  -- Calculate combined loss
  local loss = loss_real + loss_fake + loss_info

  return loss, disc_grad_params
end

local cosine_distance=function(feature_cache,k)
    local result=torch.Tensor(sum_score_D:size()):fill(0):cuda()
    for i=1,ngen do
        if i==k then
            goto continue
        else
            result=result+G.cosine:forward({ feature_cache[k],feature_cache[i]})
        end
        ::continue::
    end
    return result
end


-- Calculate outputs and gradients for the generator
local do_generator_step = function(new_params)
  if new_params ~= gen_params then
    gen_params:copy(new_params)
  end
  local gen_loss=0
  gen_grad_params:zero()
  for i=1,ngen do   
     fake_input:resizeAs(G['generator'..i].output):copy(G['generator'..i].output)
     local dbodyout = discriminator_body:forward(fake_input)

     local dheadout = discriminator_head:forward(dbodyout)

     disc_target:fill(real_label)
     local dheadout = discriminator_head.output
     local dbodyout = discriminator_body.output
     gen_loss = gen_loss+disc_head_criterion:forward(dheadout, disc_target)
     local zero_batch=torch.Tensor(sum_score_D:size()):zero():cuda()
     local diff=ngen*score_D_cache[i]-sum_score_D-cosine_distance(feature_cache,i)
     diff=diff:cuda()
     diff=-diff/(ngen-1)
     local relu_diff=G.relu:forward(diff)
     relu_diff=relu_diff:cuda()
    
     gen_loss=gen_loss+lambda_compete*compete_criterion:forward(relu_diff,zero_batch)
     local compete_df_do=G.relu:backward(diff,compete_criterion:backward(relu_diff,zero_batch)) 
     compete_df_do=lambda_compete*compete_df_do

     local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
     local dloss_ddheadin = discriminator_head:updateGradInput(dbodyout, dloss_ddheadout + compete_df_do)
  
     local iheadout = info_head:forward(dbodyout)
     local info_target = gen_inputs[i]:narrow(2, 1, n_salient_vars)
     local dloss_diheadout = info_head_criterion:updateGradInput(iheadout, info_target)
     dloss_diheadout:mul(info_regularisation_coefficient)
     local dloss_diheadin = info_head:updateGradInput(dbodyout, dloss_diheadout)
     local dloss_dgout = discriminator_body:updateGradInput(fake_input, dloss_ddheadin + dloss_diheadin  )
     G['generator'..i]:backward(gen_inputs[i], dloss_dgout)
  end
  gen_loss_meter:add(gen_loss)


  return gen_loss, gen_grad_params
end

-- Discriminator optimiser
local disc_optimiser = {
  method = optim.adam,
  config = {
    learningRate = disc_learning_rate,
    beta1 = 0.5
  },
  state = {}
}

-- Generator optimiser
local gen_optimiser = {
  method = optim.adam,
  config = {
    learningRate = gen_learning_rate,
    beta1 = 0.5
  },
  state = {}
}

-- Takes a tensor containing many images and uses them as tiles to create one
-- big image. Assumes row-major order.
local function tile_images(images, rows, cols)
  local tiled = torch.Tensor(images:size(2), images:size(3) * rows, images:size(4) * cols)
  tiled:zero()
  for i = 1, math.min(images:size(1), rows * cols) do
    local col = (i - 1) % cols
    local row = math.floor((i - 1) / cols)
    tiled
      :narrow(2, row * images:size(3) + 1, images:size(3))
      :narrow(3, col * images:size(4) + 1, images:size(4))
      :copy(images[i])
  end
  return tiled
end

-- Constant noise for each row of the fake input visualisation
local constant_noise = torch.CudaTensor(5, n_gen_inputs)
dist:sample(constant_noise:narrow(2, 1, n_salient_vars), dist.prior_params)

--local iter_inst = train_iter()

-- Training loop
for epoch = opts.start_epoch, n_epochs do
  fake_loss_meter:reset()
  info_loss_meter:reset()
  real_loss_meter:reset()
  gen_loss_meter:reset()
  time_meter:reset()

  -- Do training iterations for the epoch
  for iteration = 1, n_updates_per_epoch do
    --local sample = iter_inst()

    --if not sample or sample.input:size(1) < batch_size then
      -- Restart iterator
    --  iter_inst = train_iter()
    --  sample = iter_inst()
    --end

    -- Copy real inputs from the dataset onto the GPU
    --input = sample.input:narrow(3, 3, 28):narrow(4, 3, 28)
    input=data:getBatch()
    real_input:resize(input:size()):copy(input)

    -- Update the discriminator network
    disc_optimiser.method(
      do_discriminator_step,
      disc_params,
      disc_optimiser.config,
      disc_optimiser.state
    )
    -- Update the generator network
    gen_optimiser.method(
      do_generator_step,
      gen_params,
      gen_optimiser.config,
      gen_optimiser.state
    )
  end

  -- Generate fake images. Noise varies across rows (vertically),
  -- category varies across columns (horizontally).
  local rows = constant_noise:size(1)
  gen_input:resize(50, n_gen_inputs)
  local gen_input_view = gen_input:view(5, 10, n_gen_inputs)

  for i=1,ngen do
    G['generator'..i]:evaluate()
    dist:sample(constant_noise:narrow(2, 1, n_salient_vars), dist.prior_params)
    for k=1,opts.n_sets_categorical do
     for col = 1, 10 do
       local col_tensor = gen_input_view:select(2, col)
       col_tensor:copy(constant_noise)

       local category = col
       for row = 1, rows do
         -- Vary c1 across columns
         col_tensor[{row, {1+(k-1)*10, k*10}}]:zero()
         col_tensor[{row, (k-1)*10+category}] = 1
       end
       local images_varying_c = tile_images(G['generator'..i]:forward(gen_input):float(), 5, 10)
       local image_dir = pl.path.join('out', exp_name ,'images','dc'..k)
       pl.dir.makepath(image_dir)
 
       image.save(
      pl.path.join(image_dir, string.format('%01dvarying_dc_%04d.png',i, epoch)),
      images_varying_c)
     end
   end 
   -- for col = 1, 10 do
   --   local col_tensor = gen_input_view:select(2, col)
   --   col_tensor:copy(constant_noise)
   --   local category=col
   --   for row = 1, rows do
   --     -- Use different c1 for each row
   --     col_tensor[{row, {11, 20}}]:zero()
   --     col_tensor[{row, category+10}] = 1
   --     -- Vary c2 from -2 to 2 across columns
   --     --col_tensor[{row, {11, 11}}]:fill((col - 5.5) / 2.25)
   --   end
   -- end
   -- local images_varying_c2 = tile_images(G['generator'..i]:forward(gen_input):float(), 5, 10)
    
   -- image.save(
   -- pl.path.join(image_dir, string.format('%01dvarying_dc2_%04d.png',i, epoch)),
   -- images_varying_c2)
  end
  for i=1,ngen do
    G['generator'..i]:training()
  end
  -- Update log
  log:set{
    epoch = epoch,
    fake_loss = fake_loss_meter:value(),
    info_loss = info_loss_meter:value(),
    real_loss = real_loss_meter:value(),
    gen_loss = gen_loss_meter:value(),
    time = time_meter:value()
  }
  log:flush()

  -- Save outputs

  local model_dir = pl.path.join('out',exp_name, 'models')
  pl.dir.makepath(model_dir)

  discriminator:clearState()
  local model_disc_file = pl.path.join(model_dir, 'infogan_disc.t7')
  torch.save(model_disc_file, discriminator)

  --generator:clearState()
  local model_gen_file = pl.path.join(model_dir, 'infogan_gen.t7')
  torch.save(model_gen_file, {G=G})


  -- Checkpoint the networks every 10 epochs
  if epoch % 10 == 0 then
    pl.file.copy(model_disc_file,
      pl.path.join(model_dir, string.format('infogan_disc_%04d.t7', epoch)))
    pl.file.copy(model_gen_file,
      pl.path.join(model_dir, string.format('infogan_gen_%04d.t7', epoch)))
  end
end
