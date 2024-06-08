{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}

module MTCNN where

import GHC.Generics
import Torch
import Torch.Autograd
import Torch.NN
import qualified Torch.Functional as F
import Torch.Tensor
import qualified Torch.Functional.Internal as FI

-- PRelu
data PRelu = PRelu
  {
    prelu_weight :: Parameter
  }
  deriving (Generic, Show, Parameterized)

data PReluSpec = PReluSpec
  {
    numChannels :: Int
  }
  deriving (Show, Eq)

preluForward :: PRelu -> Tensor -> Tensor
preluForward prelu input = FI.prelu input w
  where
    w = toDependent (prelu_weight prelu)

instance Randomizable PReluSpec PRelu where
  sample PReluSpec {..} = do
    weight <- makeIndependent =<< pure (asTensor (replicate numChannels 0.25 :: [Float]))
    return $ PRelu weight

-- PNet
data PNetBB = PNetBB
  {
    pnet_c1    :: Conv2d,
    pnet_pr1   :: PRelu,
    pnet_c2    :: Conv2d,
    pnet_pr2   :: PRelu,
    pnet_c3    :: Conv2d,
    pnet_pr3   :: PRelu,
    pnet_c4_1  :: Conv2d,
    pnet_c4_2  :: Conv2d
  }
  deriving (Generic, Show)

data PNetBBSpec = PNetBBSpec
  {
    pnet_conv1   :: Conv2dSpec,
    pnet_prelu1  :: PReluSpec,
    pnet_conv2   :: Conv2dSpec,
    pnet_prelu2  :: PReluSpec,
    pnet_conv3   :: Conv2dSpec,
    pnet_prelu3  :: PReluSpec,
    pnet_conv4_1 :: Conv2dSpec,
    pnet_conv4_2 :: Conv2dSpec
  }
  deriving (Show, Eq)

pnetBackBoneSpec = 
  PNetBBSpec
    (Conv2dSpec 3 10 3 3)
    (PReluSpec 10)
    (Conv2dSpec 10 16 3 3)
    (PReluSpec 16)
    (Conv2dSpec 16 32 3 3)
    (PReluSpec 32)
    (Conv2dSpec 32 2 1 1)
    (Conv2dSpec 32 4 1 1)

instance Parameterized PNetBB

instance Randomizable PNetBBSpec PNetBB where
  sample PNetBBSpec {..} =
    PNetBB
      <$> sample pnet_conv1
      <*> sample pnet_prelu1
      <*> sample pnet_conv2
      <*> sample pnet_prelu2
      <*> sample pnet_conv3
      <*> sample pnet_prelu3
      <*> sample pnet_conv4_1
      <*> sample pnet_conv4_2

pnetBBForward :: PNetBB -> Tensor -> (Tensor, Tensor)
pnetBBForward PNetBB {..} input = (b, a)
  where
    b = conv2dForward   pnet_c4_2 (1, 1) (0, 0) x
    a = softmax (Dim 1) 
        . conv2dForward pnet_c4_1 (1, 1) (0, 0)
         $ x
    x = preluForward      pnet_pr3
          . conv2dForward pnet_c3 (1, 1) (0, 0)
          . preluForward  pnet_pr2
          . conv2dForward pnet_c2 (1, 1) (0, 0)
          . maxPool2d (2, 2) (2, 2) (0, 0) (1, 1) Ceil
          . preluForward  pnet_pr1
          . conv2dForward pnet_c1 (1, 1) (0, 0)
          $ input

-- RNet
data RNetBB = RNetBB
  {
    rnet_c1    :: Conv2d,
    rnet_pr1   :: PRelu,
    rnet_c2    :: Conv2d,
    rnet_pr2   :: PRelu,
    rnet_c3    :: Conv2d,
    rnet_pr3   :: PRelu,
    rnet_d4    :: Linear,
    rnet_pr4   :: PRelu,
    rnet_d5_1  :: Linear,
    rnet_d5_2  :: Linear
  }
  deriving (Generic, Show)

instance Parameterized RNetBB

data RNetBBSpec = RNetBBSpec
  {
    rnet_conv1     :: Conv2dSpec,
    rnet_prelu1    :: PReluSpec,
    rnet_conv2     :: Conv2dSpec,
    rnet_prelu2    :: PReluSpec,
    rnet_conv3     :: Conv2dSpec,
    rnet_prelu3    :: PReluSpec,
    rnet_dence4    :: LinearSpec,
    rnet_prelu4    :: PReluSpec,
    rnet_dence5_1  :: LinearSpec,
    rnet_dence5_2  :: LinearSpec
  }
  deriving (Show, Eq)

rnetBackBoneSpec = 
  RNetBBSpec
    (Conv2dSpec 3 28 3 3)
    (PReluSpec 28)
    (Conv2dSpec 28 48 3 3)
    (PReluSpec 48)
    (Conv2dSpec 48 64 2 2)
    (PReluSpec 64)
    (LinearSpec 576 128)
    (PReluSpec 128)
    (LinearSpec 128 2)
    (LinearSpec 128 4)

instance Randomizable RNetBBSpec RNetBB where
  sample RNetBBSpec {..} =
    RNetBB
      <$> sample rnet_conv1
      <*> sample rnet_prelu1
      <*> sample rnet_conv2
      <*> sample rnet_prelu2
      <*> sample rnet_conv3
      <*> sample rnet_prelu3
      <*> sample rnet_dence4
      <*> sample rnet_prelu4
      <*> sample rnet_dence5_1
      <*> sample rnet_dence5_2

view_for_rnet :: Tensor -> Tensor
view_for_rnet input = view [ (shape input) !! 0, -1] input

rnetBBForward :: RNetBB -> Tensor -> (Tensor, Tensor)
rnetBBForward RNetBB {..} input = (b, a)
  where
    a = softmax (Dim 1) 
        . linear rnet_d5_1
        $ x
    b = linear rnet_d5_2
        $ x
    x = preluForward      rnet_pr4
          . linear        rnet_d4 
          . view_for_rnet
          . contiguous
          . F.permute [0, 3, 2, 1]
          . preluForward  rnet_pr3
          . conv2dForward rnet_c3 (1, 1) (0, 0)
          . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) Ceil
          . preluForward  rnet_pr2
          . conv2dForward rnet_c2 (1, 1) (0, 0)
          . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) Ceil
          . preluForward  rnet_pr1
          . conv2dForward rnet_c1 (1, 1) (0, 0)
          $ input

-- ONet
data ONetBB = ONetBB
  {
    onet_c1    :: Conv2d,
    onet_pr1   :: PRelu,
    onet_c2    :: Conv2d,
    onet_pr2   :: PRelu,
    onet_c3    :: Conv2d,
    onet_pr3   :: PRelu,
    onet_c4    :: Conv2d,
    onet_pr4   :: PRelu,
    onet_d5    :: Linear,
    onet_pr5   :: PRelu,
    onet_d6_1  :: Linear,
    onet_d6_2  :: Linear,
    onet_d6_3  :: Linear
  }
  deriving (Generic, Show)

instance Parameterized ONetBB

data ONetBBSpec = ONetBBSpec
  {
    onet_conv1     :: Conv2dSpec,
    onet_prelu1    :: PReluSpec,
    onet_conv2     :: Conv2dSpec,
    onet_prelu2    :: PReluSpec,
    onet_conv3     :: Conv2dSpec,
    onet_prelu3    :: PReluSpec,
    onet_conv4     :: Conv2dSpec,
    onet_prelu4    :: PReluSpec,
    onet_dence5    :: LinearSpec,
    onet_prelu5    :: PReluSpec,
    onet_dence6_1  :: LinearSpec,
    onet_dence6_2  :: LinearSpec,
    onet_dence6_3  :: LinearSpec
  }
  deriving (Show, Eq)


onetBackBoneSpec = 
  ONetBBSpec
    (Conv2dSpec 3 32 3 3)
    (PReluSpec 32)
    (Conv2dSpec 32 64 3 3)
    (PReluSpec 64)
    (Conv2dSpec 64 64 3 3)
    (PReluSpec 64)
    (Conv2dSpec 64 128 2 2)
    (PReluSpec 128)
    (LinearSpec 1152 256)
    (PReluSpec 256)
    (LinearSpec 256 2)
    (LinearSpec 256 4)
    (LinearSpec 256 10)


instance Randomizable ONetBBSpec ONetBB where
  sample ONetBBSpec {..} =
    ONetBB
      <$> sample onet_conv1
      <*> sample onet_prelu1
      <*> sample onet_conv2
      <*> sample onet_prelu2
      <*> sample onet_conv3
      <*> sample onet_prelu3
      <*> sample onet_conv4
      <*> sample onet_prelu4
      <*> sample onet_dence5
      <*> sample onet_prelu5
      <*> sample onet_dence6_1
      <*> sample onet_dence6_2
      <*> sample onet_dence6_3

onetBBForward :: ONetBB -> Tensor -> (Tensor, Tensor, Tensor)
onetBBForward ONetBB {..} input = (b, c, a)
  where
    a = softmax (Dim 1) 
        . linear onet_d6_3
        $ x
    b = linear onet_d6_2
        $ x
    c = linear onet_d6_1
        $ x
    x = preluForward      onet_pr5
          . linear        onet_d5
          . view_for_rnet
          . contiguous
          . F.permute [0, 3, 2, 1]
          . preluForward  onet_pr4
          . conv2dForward onet_c4 (1, 1) (0, 0)
          . maxPool2d (2, 2) (2, 2) (0, 0) (1, 1) Ceil
          . preluForward  onet_pr3
          . conv2dForward onet_c3 (1, 1) (0, 0)
          . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) Ceil
          . preluForward  onet_pr2
          . conv2dForward onet_c2 (1, 1) (0, 0)
          . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) Ceil
          . preluForward  onet_pr1
          . conv2dForward onet_c1 (1, 1) (0, 0)
          $ input