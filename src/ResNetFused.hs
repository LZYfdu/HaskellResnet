{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}

module ResNetFused where

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

{-
-- Conv2dwithStrideAndPadding
data Conv2dwithSP = Conv2dwithSP 
  {
    conv2d_with_sp_c1 :: Conv2d
  }
  deriving (Show, Generic, Parameterized)

data Conv2dwithSPSpec = Conv2dwithSPSpec 
  {
    conv2dwithSPinputC :: Int,
    conv2dwithSPoutputC :: Int,
    conv2dwithSPkernelHeight :: Int,
    conv2dwithSPkernelWidth :: Int,
    conv2dwithSPstride :: Int,
    conv2dwithSPpadding :: Int,
  }
  deriving (Show, Eq)

instance Randomizable Conv2dwithSPSpec Conv2dwithSP where
  sample Conv2dwithSPSpec {..} =
    Conv2dwithSP
      <$> sample (Conv2dSpec conv2dwithSPinputC conv2dwithSPoutputC conv2dwithSPkernelHeight conv2dwithSPkernelWidth)

conv2dWSPForward :: Conv2dwithSP -> Tensor -> Tensor
conv2dWSPForward Conv2dwithSP {..} input = conv2dForward conv2d_with_sp_c1 (conv2dwithSPstride, conv2dwithSPstride) (conv2dwithSPpadding, conv2dwithSPpadding) input
-}


-- Bottleneck
data BottleneckBB = BottleneckBB 
  {
    bneck_c1    :: Conv2d,
    bneck_c2    :: Conv2d,
    bneck_c3    :: Conv2d
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
      <$> sample (Conv2dSpec bneck_in_places bneck_places 1 1)
      <*> sample (Conv2dSpec bneck_places bneck_places 3 3)
      <*> sample (Conv2dSpec bneck_places (bneck_places * bneck_expansion) 1 1)

bottleneckBBForward :: BottleneckBB -> Int -> Tensor -> Tensor
bottleneckBBForward BottleneckBB {..} stride input = F.relu (F.add input out)
  where
    out = conv2dForward bneck_c3 (1, 1) (0, 0)
          . F.relu
          . conv2dForward bneck_c2 (stride, stride) (1, 1)
          . F.relu
          . conv2dForward bneck_c1 (1, 1) (0, 0)
          $ input

-- BottleneckDownsample
data BottleneckDownsampleBB = BottleneckDownsampleBB 
  {
    bneck_down_c1    :: Conv2d,
    bneck_down_c2    :: Conv2d,
    bneck_down_c3    :: Conv2d,
    bneck_down_c4    :: Conv2d
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
      <$> sample (Conv2dSpec bneck_down_in_places bneck_down_places 1 1)
      <*> sample (Conv2dSpec bneck_down_places bneck_down_places 3 3)
      <*> sample (Conv2dSpec bneck_down_places (bneck_down_places * bneck_down_expansion) 1 1)
      <*> sample (Conv2dSpec bneck_down_in_places (bneck_down_places * bneck_down_expansion) 1 1)

bottleneckDownsampleBBForward :: BottleneckDownsampleBB -> Int -> Tensor -> Tensor
bottleneckDownsampleBBForward BottleneckDownsampleBB {..} stride input = F.relu (F.add res out)
  where
    res = conv2dForward bneck_down_c4 (stride, stride) (0, 0)
          $ input
    out = conv2dForward bneck_down_c3 (1, 1) (0, 0)
          . F.relu
          . conv2dForward bneck_down_c2 (stride, stride) (1, 1)
          . F.relu
          . conv2dForward bneck_down_c1 (1, 1) (0, 0)
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

makedLayerBBForward :: MakedLayer -> Int -> Tensor -> Tensor
makedLayerBBForward MakedLayer {..} stride input = forwardRemain (bottleneckDownsampleBBForward mklayer_btn1 stride input) mklayer_btns
  where
    forwardRemain i_tensor [] = i_tensor
    forwardRemain i_tensor (bneck:btns) = forwardRemain (bottleneckBBForward bneck 1 i_tensor) btns

-- ResNet
data ResNet = ResNet 
  {
    resnet_c1     :: Conv2d,
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
      <$> sample (Conv2dSpec 3 64 7 7)
      <*> sample (MakedLayerSpec 64 64 (resnet_blocks !! 0) 1)
      <*> sample (MakedLayerSpec 256 128 (resnet_blocks !! 1) 2)
      <*> sample (MakedLayerSpec 512 256 (resnet_blocks !! 2) 2)
      <*> sample (MakedLayerSpec 1024 512 (resnet_blocks !! 3) 2)
      <*> sample (LinearSpec 2048 resnet_num_classes)

resnetForward :: ResNet -> Tensor -> Tensor
resnetForward ResNet {..} input =
  linear resnet_fc
  . F.view [ shape input !! 0, -1]
  . F.adaptiveAvgPool2d (1, 1)
  . makedLayerBBForward resnet_layer4 2
  . makedLayerBBForward resnet_layer3 2
  . makedLayerBBForward resnet_layer2 2
  . makedLayerBBForward resnet_layer1 1
  . F.maxPool2d (3, 3) (2, 2) (1, 1) (1, 1) Floor
  . F.relu
  . conv2dForward resnet_c1 (2, 2) (3, 3)
  $ input