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
local model_builder = require('classifying_dcgan_model_builder')
unpack=unpack or table.unpack
--- OPTIONS ---

local opts = pl.lapp [[
Trains an InfoGAN network
  --epochs (default 1000) Number of training epochs
  --updates-per-epoch (default 100) Number of batches per epoch
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
  --n-sets-categorical (default 3) The sets of categorical variables
  --gpu (default 1) Which GPU to use
  --continue-train (default true)
  --start-epoch (default 1)
  --out-size (default 64)
  --trained-generator (default 'infogan_gen.t7')
  --save-name (default '/home/torrvision/data/inception')   name of the file saved
  --nsamples (default 50000) 
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

local model_dir = pl.path.join('out',exp_name, 'models')


local model_gen_file = pl.path.join(model_dir, 'infogan_gen.t7')

G=torch.load(model_gen_file).G

gen_input=torch.CudaTensor(batch_size, n_gen_inputs)
paths.mkdir(opts.save_name)
for iter=1,opts.nsamples,opts.batch_size do
    for i=1,ngen do
     	dist:sample(gen_input:narrow(2, 1, n_salient_vars), dist.prior_params)
     	gen_input:narrow(2, n_salient_vars + 1, n_noise_vars):normal(0, 1)
        paths.mkdir(opts.save_name..'/gen_'..tostring(i))
        local images_Gi = G['generator'..i]:forward(gen_input)
        --print('Images size: ', images_Gi:size(1)..' x '..images_Gi:size(2) ..' x '..images_Gi:size(3)..' x '..images_Gi:size(4))
        images_Gi:add(1):mul(0.5)
        --print('Min, Max, Mean, Stdv', images_G1:min(), images_G1:max(), images_G1:mean(), images_G1:std())

        for j=1,opts.batch_size do
            image.save(opts.save_name .. '/gen_'..tostring(i).. '/' .. tostring(iter+j-1) ..  '.png', image.toDisplayTensor(images_Gi[j]))
        end
        --print('Saved image to: ', opt.name .. 'images_G1.png')
    end
end
