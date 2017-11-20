require('cudnn')
local nninit = require('nninit')

local model_builder = {}

local Seq = nn.Sequential
local ReLU = cudnn.ReLU

local function SpatBatchNorm(n_outputs)
  return nn.SpatialBatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 1.0, 0.02) -- Gamma
    :init('bias', nninit.constant, 0)         -- Beta
end

local function BatchNorm(n_outputs)
  return nn.BatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 1.0, 0.02) -- Gamma
    :init('bias', nninit.constant, 0)         -- Beta
end

local function Conv(...)
  local conv = cudnn.SpatialConvolution(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)

  -- Use deterministic algorithms for convolution
  conv:setMode(
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')

  return conv
end

local function FullConv(...)
  local conv = cudnn.SpatialFullConvolution(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)

  -- Use deterministic algorithms for convolution
  conv:setMode(
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')

  return conv
end

local function LeakyReLU(leakiness, in_place)
  leakiness = leakiness or 0.01
  in_place = in_place == nil and true or in_place
  return nn.LeakyReLU(leakiness, in_place)
end

local function Linear(...)
  return nn.Linear(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)
end

function model_builder.build_infogan(n_gen_inputs, n_salient_params,opt)
--  local generator = Seq()
--    -- n_gen_inputs
--    :add(Linear(n_gen_inputs, 1024))
--    :add(BatchNorm(1024))
--    :add(ReLU(true))
--    -- 1024
--    :add(Linear(1024, 128 * 7 * 7))
--    :add(BatchNorm(128 * 7 * 7))
--    :add(ReLU(true))
--    :add(nn.Reshape(128, 7, 7))
--    -- 128 x 7 x 7
--    :add(FullConv(128, 64, 4,4, 2,2, 1,1))
--    :add(SpatBatchNorm(64))
--    :add(ReLU(true))
--    -- 64 x 14 x 14
--    :add(FullConv(64, 1, 4,4, 2,2, 1,1))
--    :add(nn.Sigmoid())
--    -- 1 x 28 x 28
    local SpatialBatchNormalization = nn.SpatialBatchNormalization
    local SpatialConvolution = nn.SpatialConvolution
    local SpatialFullConvolution = nn.SpatialFullConvolution


    local nc=3
    local nz=n_gen_inputs
    local ndf=opt.ndf
    local ngf=opt.ngf
    local netG = nn.Sequential()
    netG:add(nn.Reshape(nz,1,1))
    -- input is Z, going into a convolution
    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    -- state size: (ngf*8) x 4 x 4
    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 8
    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
    -- state size: (ngf*2) x 16 x 16
    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    if opt.out_size==64 then
         netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
         -- state size: (ngf) x 32 x 32
         netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
    end
    netG:add(nn.Tanh())
    -- state size: (nc) x 64 x 64

--  local discriminator_body = Seq()
--    -- 1 x 28 x 28
--    :add(Conv(1, 64, 4,4, 2,2, 1,1))
--    :add(LeakyReLU())
--    -- 64 x 14 x 14
--    :add(Conv(64, 128, 4,4, 2,2, 1,1))
--    :add(SpatBatchNorm(128))
--    :add(LeakyReLU())
--    -- 128 x 7 x 7
--    :add(nn.Reshape(128 * 7 * 7))
--    :add(Linear(128 * 7 * 7, 1024))
--    :add(BatchNorm(1024))
--    :add(LeakyReLU())
--    -- 1024
    local SpatialBatchNormalization = nn.SpatialBatchNormalization
    local SpatialConvolution = nn.SpatialConvolution
    local SpatialFullConvolution = nn.SpatialFullConvolution


    local netD = nn.Sequential()
    if opt.out_size==64 then 
        -- input is (nc) x 64 x 64
        netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf) x 32 x 32
        netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    else if opt.out_size==32 then
         --netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
         --netD:add(nn.LeakyReLU(0.2, true))
         -- state size: (ndf) x 32 x 32
         netD:add(SpatialConvolution(nc, ndf * 2, 4, 4, 2, 2, 1, 1))
    end
    netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 8 x 8
    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4


  local discriminator_head =nn.Sequential()
    ---- 1024
    --:add(Linear(1024, 1))
    --:add(nn.Sigmoid())
    ---- 1
    discriminator_head:add(SpatialConvolution(ndf * 8, 1, 4, 4))
    discriminator_head:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
    discriminator_head:add(nn.View(1):setNumInputDims(3))
    -- state size: 1


  local info_head = nn.Sequential()
    ---- 1024
    --:add(Linear(1024, 128))
    --:add(BatchNorm(128))
    --:add(LeakyReLU())
    ---- 128
    --:add(Linear(128, n_salient_params))
    ---- n_salient_params
    info_head:add(nn.Reshape(ndf*8*4*4))
    info_head:add(nn.Linear(ndf*8*4*4,128))
    info_head:add(BatchNorm(128))
    info_head:add(LeakyReLU())
    -- 128
    info_head:add(nn.Linear(128,n_salient_params))

  return netG, netD, discriminator_head, info_head
end

function model_builder.build_infogan_heads(n_gen_inputs, n_salient_params,opt)
--  local generator = Seq()
--    -- n_gen_inputs
--    :add(Linear(n_gen_inputs, 1024))
--    :add(BatchNorm(1024))
--    :add(ReLU(true))
--    -- 1024
--    :add(Linear(1024, 128 * 7 * 7))
--    :add(BatchNorm(128 * 7 * 7))
--    :add(ReLU(true))
--    :add(nn.Reshape(128, 7, 7))
--    -- 128 x 7 x 7
--    :add(FullConv(128, 64, 4,4, 2,2, 1,1))
--    :add(SpatBatchNorm(64))
--    :add(ReLU(true))
--    -- 64 x 14 x 14
--    :add(FullConv(64, 1, 4,4, 2,2, 1,1))
--    :add(nn.Sigmoid())
--    -- 1 x 28 x 28
    local G={}
    local SpatialBatchNormalization = nn.SpatialBatchNormalization
    local SpatialConvolution = nn.SpatialConvolution
    local SpatialFullConvolution = nn.SpatialFullConvolution


    local nc=3
    local nz=n_gen_inputs
    local ndf=opt.ndf
    local ngf=opt.ngf
    local netG = nn.Sequential()
    netG:add(nn.Reshape(nz,1,1))
    -- input is Z, going into a convolution
    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    -- state size: (ngf*8) x 4 x 4
    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 8
    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
    -- state size: (ngf*2) x 16 x 16
    for i=1,opt.ngen do
        G['netG'..i]=nn.Sequential()
        if i==1 then 
            G['netG'..i]:add(netG)
        else
            G['netG'..i]:add(netG:clone('weight','bias','gradWeight','gradBias'))
        end
        G['netG'..i]:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
        if opt.out_size==64 then
            G['netG'..i]:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
            -- state size: (ngf) x 32 x 32
            G['netG'..i]:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
        end
        G['netG'..i]:add(nn.Tanh())
    end
    -- state size: (nc) x 64 x 64

--  local discriminator_body = Seq()
--    -- 1 x 28 x 28
--    :add(Conv(1, 64, 4,4, 2,2, 1,1))
--    :add(LeakyReLU())
--    -- 64 x 14 x 14
--    :add(Conv(64, 128, 4,4, 2,2, 1,1))
--    :add(SpatBatchNorm(128))
--    :add(LeakyReLU())
--    -- 128 x 7 x 7
--    :add(nn.Reshape(128 * 7 * 7))
--    :add(Linear(128 * 7 * 7, 1024))
--    :add(BatchNorm(1024))
--    :add(LeakyReLU())
--    -- 1024
    local SpatialBatchNormalization = nn.SpatialBatchNormalization
    local SpatialConvolution = nn.SpatialConvolution
    local SpatialFullConvolution = nn.SpatialFullConvolution


    local netD = nn.Sequential()
    if opt.out_size==64 then 
        -- input is (nc) x 64 x 64
        netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf) x 32 x 32
        netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    else if opt.out_size==32 then
         --netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
         --netD:add(nn.LeakyReLU(0.2, true))
         -- state size: (ndf) x 32 x 32
         netD:add(SpatialConvolution(nc, ndf * 2, 4, 4, 2, 2, 1, 1))
    end
    netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 8 x 8
    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4


  local discriminator_head =nn.Sequential()
    ---- 1024
    --:add(Linear(1024, 1))
    --:add(nn.Sigmoid())
    ---- 1
    discriminator_head:add(SpatialConvolution(ndf * 8, 1, 4, 4))
    discriminator_head:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
    discriminator_head:add(nn.View(1):setNumInputDims(3))
    -- state size: 1


  local info_head = nn.Sequential()
    ---- 1024
    --:add(Linear(1024, 128))
    --:add(BatchNorm(128))
    --:add(LeakyReLU())
    ---- 128
    --:add(Linear(128, n_salient_params))
    ---- n_salient_params
    info_head:add(nn.Reshape(ndf*8*4*4))
    info_head:add(nn.Linear(ndf*8*4*4,128))
    info_head:add(BatchNorm(128))
    info_head:add(LeakyReLU())
    -- 128
    info_head:add(nn.Linear(128,n_salient_params))


  return G, netD, discriminator_head, info_head
end

return model_builder
