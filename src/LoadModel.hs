module LoadModel where

import MTCNN(PNetBB, RNetBB, ONetBB, pnetBackBoneSpec, rnetBackBoneSpec, onetBackBoneSpec)
import ResNetFused
import Torch.Script
import Torch.Autograd
import Torch.NN

loadModel :: (Randomizable s n, Parameterized n) => FilePath -> s -> IO n
loadModel path specNet = do
  model <- loadScript WithoutRequiredGrad path
  let params = map IndependentTensor $ getParameters model
  net_init <- sample specNet
  return $ replaceParameters net_init params

  
resnet50Spec = ResNetSpec [3, 4, 6, 3] 1000

loadResNet50 :: FilePath -> IO ResNet
loadResNet50 path = loadModel path resnet50Spec

{-
loadPNet :: FilePath -> IO PNetBB
loadPNet path = loadModel path pnetBackBoneSpec

loadRNet :: FilePath -> IO RNetBB
loadRNet path = loadModel path rnetBackBoneSpec

loadONet :: FilePath -> IO ONetBB
loadONet path = loadModel path onetBackBoneSpec
-}

