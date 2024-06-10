{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}

module ResNet where

import GHC.Generics
import Torch
import Torch.Autograd
import Torch.NN
import qualified Torch.Functional as F
import Torch.Tensor
import qualified Torch.Functional.Internal as FI
import Torch.Initializers

import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (replicateM)

-- Conv2dWithoutBias
data Conv2dWithoutBias = Conv2dWithoutBias
  {
    conv2d_weight :: Parameter
  }
  deriving (Show, Generic, Parameterized)

data Conv2dWithoutBiasSpec = Conv2dWithoutBiasSpec 
  {
    conv2dnb_in_channels  :: Int,
    conv2dnb_out_channels :: Int,
    conv2dnb_kernelHeight :: Int,
    conv2dnb_kernelWidth  :: Int
  }
  deriving (Show, Eq)

instance Randomizable Conv2dWithoutBiasSpec Conv2dWithoutBias where
  sample Conv2dWithoutBiasSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ conv2dnb_in_channels,
            conv2dnb_out_channels,
            conv2dnb_kernelHeight,
            conv2dnb_kernelWidth
          ]
    return $ Conv2dWithoutBias w

conv2dWithoutBiasForward :: Conv2dWithoutBias -> (Int, Int) -> (Int, Int) -> Tensor -> Tensor
conv2dWithoutBiasForward conv_nb = F.conv2d' w b 
  where
    w = toDependent $ conv2d_weight conv_nb
    b = zeros' [outputChannel]
    outputChannel = shape w !! 0

-- MyBN
data MyBNBB = MyBNBB
  {
    mybn_weight      :: Parameter,
    mybn_bias        :: Parameter,
    mybn_runningMean :: Parameter,
    mybn_runningVar  :: Parameter
  }
  deriving (Show, Generic, Parameterized)

data MyBNBBSpec = MyBNBBSpec 
  {
    mybn_numFeatures :: Int
  }
  deriving (Show, Eq)

instance Randomizable MyBNBBSpec MyBNBB where
  sample MyBNBBSpec {..} = do
    w <- makeIndependent (ones' [mybn_numFeatures])
    b <- makeIndependent (zeros' [mybn_numFeatures])
    running_mean <- makeIndependentWithRequiresGrad (zeros' [mybn_numFeatures]) False
    running_var <- makeIndependentWithRequiresGrad (ones' [mybn_numFeatures]) False
    return $ MyBNBB w b running_mean running_var

mybnforwardIO :: MyBNBB -> Bool -> Double -> Double -> Tensor -> IO Tensor
mybnforwardIO bn train momentum eps input = 
  F.batchNormIO
    (toDependent $ mybn_weight bn)
    (toDependent $ mybn_bias bn)
    (MutableTensor $ toDependent $ mybn_runningMean bn)
    (MutableTensor $ toDependent $ mybn_runningVar bn)
    train
    momentum
    eps
    input

mybatchNormForward :: MyBNBB -> Tensor -> Tensor
mybatchNormForward bn input = unsafePerformIO $ do
    output <- mybnforwardIO bn False 0.1 1e-5 input
    return output

{-
-- Bottleneck
data BottleneckBB = BottleneckBB 
  {
    bneck_c1    :: Conv2dWithoutBias,
    bneck_bn1   :: MyBNBB,
    bneck_c2    :: Conv2dWithoutBias,
    bneck_bn2   :: MyBNBB,
    bneck_c3    :: Conv2dWithoutBias,
    bneck_bn3   :: MyBNBB
  }
  deriving (Show, Generic)

instance Parameterized BottleneckBB

data BottleneckBBSpec = BottleneckBBSpec 
  {
    bneck_in_places        :: Int,
    bneck_places           :: Int,
    bneck_stride           :: Int,
    bneck_expansion        :: Int
  }
  deriving (Show, Eq)

instance Randomizable BottleneckBBSpec BottleneckBB where
  sample BottleneckBBSpec {..} =
    BottleneckBB
      <$> sample (Conv2dWithoutBiasSpec bneck_in_places bneck_places 1 1)
      <*> sample (MyBNBBSpec bneck_places)
      <*> sample (Conv2dWithoutBiasSpec bneck_places bneck_places 3 3)
      <*> sample (MyBNBBSpec bneck_places)
      <*> sample (Conv2dWithoutBiasSpec bneck_places (bneck_places * bneck_expansion) 1 1)
      <*> sample (MyBNBBSpec (bneck_places * bneck_expansion))

bottlenekBBForward :: BottleneckBB -> Tensor -> Tensor
bottlenekBBForward BottleneckBB {..} input = F.relu (F.add input out)
  where
    out = mybatchNormForward bneck_bn3
          . conv2dWithoutBiasForward bneck_c3 (1, 1) (0, 0)
          . F.relu
          . mybatchNormForward bneck_bn2
          . conv2dWithoutBiasForward bneck_c2 (3, 3) (1, 1)
          . F.relu
          . mybatchNormForward bneck_bn1
          . conv2dWithoutBiasForward bneck_c1 (1, 1) (0, 0)
          $ input


-- BottleneckDownsample
data BottleneckDownsampleBB = BottleneckDownsampleBB 
  {
    bneck_down_c1    :: Conv2dWithoutBias,
    bneck_down_bn1   :: MyBNBB,
    bneck_down_c2    :: Conv2dWithoutBias,
    bneck_down_bn2   :: MyBNBB,
    bneck_down_c3    :: Conv2dWithoutBias,
    bneck_down_bn3   :: MyBNBB,
    bneck_down_c4    :: Conv2dWithoutBias,
    bneck_down_bn4   :: MyBNBB
  }
  deriving (Show, Generic)

instance Parameterized BottleneckDownsampleBB

data BottleneckDownsampleBBSpec = BottleneckDownsampleBBSpec 
  {
    bneck_down_in_places  :: Int,
    bneck_down_places     :: Int,
    bneck_down_stride     :: Int,
    bneck_down_expansion  :: Int
  }
  deriving (Show, Eq)


instance Randomizable BottleneckDownsampleBBSpec BottleneckDownsampleBB where
  sample BottleneckDownsampleBBSpec {..} =
    BottleneckDownsampleBB
      <$> sample (Conv2dWithoutBiasSpec bneck_down_in_places bneck_down_places 1 1)
      <*> sample (MyBNBBSpec bneck_down_places)
      <*> sample (Conv2dWithoutBiasSpec bneck_down_places bneck_down_places 3 3)
      <*> sample (MyBNBBSpec bneck_down_places)
      <*> sample (Conv2dWithoutBiasSpec bneck_down_places (bneck_down_places * bneck_down_expansion) 1 1)
      <*> sample (MyBNBBSpec (bneck_down_places * bneck_down_expansion))
      <*> sample (Conv2dWithoutBiasSpec bneck_down_in_places (bneck_down_places * bneck_down_expansion) 1 1)
      <*> sample (MyBNBBSpec (bneck_down_places * bneck_down_expansion))

bottleneckDownsampleBBForward :: BottleneckDownsampleBB -> Tensor -> Tensor
bottleneckDownsampleBBForward BottleneckDownsampleBB {..} input = F.relu (F.add res out)
  where
    res = mybatchNormForward bneck_down_bn4
          . conv2dWithoutBiasForward bneck_down_c4 (1, 1) (0, 0)
          $ input
    out = mybatchNormForward bneck_down_bn3
          . conv2dWithoutBiasForward bneck_down_c3 (1, 1) (0, 0)
          . F.relu
          . mybatchNormForward bneck_down_bn2
          . conv2dWithoutBiasForward bneck_down_c2 (3, 3) (1, 1)
          . F.relu
          . mybatchNormForward bneck_down_bn1
          . conv2dWithoutBiasForward bneck_down_c1 (1, 1) (0, 0)
          $ input


-- MakedLayer
data MakedLayer = MakedLayer 
  {
    mklayer_btn1 :: BottleneckDownsampleBB,
    mklayer_btns :: [BottleneckBB]
  }
  deriving(Generic, Show)

instance Parameterized MakedLayer

data MakedLayerSpec = MakedLayerSpec 
  {
    mklayer_in_places  :: Int,
    mklayer_places     :: Int,
    mklayer_block      :: Int,
    mklayer_stride     :: Int
  }
  deriving (Show, Eq)

instance Randomizable MakedLayerSpec MakedLayer where
  sample MakedLayerSpec {..} =
    MakedLayer 
      <$> sample (BottleneckDownsampleBBSpec mklayer_in_places mklayer_places mklayer_stride 4)
      <*> replicateM (mklayer_block - 1) (sample (BottleneckBBSpec (mklayer_places * 4) mklayer_places mklayer_stride 4))

makedLayerBBForward :: MakedLayer -> Tensor -> Tensor
makedLayerBBForward MakedLayer {..} input = foldr (.) id (map bottlenekBBForward mklayer_btns) 
                                            . bottleneckDownsampleBBForward mklayer_btn1 
                                            $ input


-- ResNet
data ResNet = ResNet 
  {
    resnet_c1     :: Conv2dWithoutBias,
    resnet_bn1    :: MyBNBB,
    resnet_layer1 :: MakedLayer,
    resnet_layer2 :: MakedLayer,
    resnet_layer3 :: MakedLayer,
    resnet_layer4 :: MakedLayer,
    resnet_fc     :: Linear
  }
  deriving (Generic, Show)

instance Parameterized ResNet

data ResNetSpec = ResNetSpec 
  {
    resnet_blocks  :: [Int],
    resnet_num_classes :: Int
  }
  deriving (Show, Eq)

instance Randomizable ResNetSpec ResNet where
  sample ResNetSpec {..} =
    ResNet
      <$> sample (Conv2dWithoutBiasSpec 3 64 7 7)
      <*> sample (MyBNBBSpec 64)
      <*> sample (MakedLayerSpec 64 64 (resnet_blocks !! 0) 1)
      <*> sample (MakedLayerSpec 256 128 (resnet_blocks !! 1) 2)
      <*> sample (MakedLayerSpec 512 256 (resnet_blocks !! 2) 2)
      <*> sample (MakedLayerSpec 1024 512 (resnet_blocks !! 3) 2)
      <*> sample (LinearSpec 2048 resnet_num_classes)

resnetForward :: ResNet -> Tensor -> Tensor
resnetForward ResNet {..} input =
  linear resnet_fc
  . F.view [ shape input !! 0, -1]
  . flip FI.adaptive_avg_pool2d (1, 1)
  . makedLayerBBForward resnet_layer4
  . makedLayerBBForward resnet_layer3
  . makedLayerBBForward resnet_layer2
  . makedLayerBBForward resnet_layer1
  . F.maxPool2d (3, 3) (2, 2) (1, 1) (1, 1) Floor
  . F.relu
  . mybatchNormForward resnet_bn1
  . conv2dWithoutBiasForward resnet_c1 (2, 2) (3, 3)
  $ input


-}