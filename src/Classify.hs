
module Classify where

import LoadModel
import Torch.Vision
import Torch.Tensor
import Torch.DType
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import Codec.Picture
import Codec.Picture.Types
import ResNetFused


getImageFrom :: Either String Tensor -> Tensor
getImageFrom (Left _) = error "Invalid image path"
getImageFrom (Right img) = img

preprocessImage :: Tensor -> Tensor
preprocessImage img = toType Float
                      . F.permute [0, 3, 1, 2]
                      $ img
{-
downScaleImage :: Image PixelRGB8 -> Int -> Int -> Image PixelRGB8
downScaleImage inputImage targetWidth targetHeight = do
  let imageWidth' = imageWidth inputImage
      imageHeight' = imageHeight inputImage
      scaleWidth = fromIntegral targetWidth / fromIntegral imageWidth'
      scaleHeight = fromIntegral targetHeight / fromIntegral imageHeight'
      newImage = generateImage
        (\x y -> let px = round (fromIntegral x / scaleWidth)
                     py = round (fromIntegral y / scaleHeight)
                 in pixelAt inputImage px py)
        targetWidth
        targetHeight
  newImage
-}

fromRGB8ToTensor :: Image PixelRGB8 -> Tensor
fromRGB8ToTensor img = preprocessImage
                       . Torch.Vision.fromDynImage
                       . Codec.Picture.ImageRGB8
                       $ img

castScalarDTwo :: (Int, Int) -> Float -> Tensor
castScalarDTwo (dt1, dt2) x = F.view [1, dt1, dt2] (asTensor (replicate dt1 $ replicate dt2 x))

genMean :: (Int,Int) -> (Float, Float, Float) -> Tensor
genMean (d1, d2) (r, g, b) = F.view [1, 3, d1, d2] (F.cat (F.Dim 0) [mr, mg, mb])
  where
    mr = castScalarDTwo (d1, d2) r
    mg = castScalarDTwo (d1, d2) g
    mb = castScalarDTwo (d1, d2) b

normalizeImage :: Tensor -> Tensor -> Tensor -> Tensor
normalizeImage img mean std = F.div (F.sub (F.divScalar (255.0 :: Float) img) mean) std

resizeAndnorm :: Image PixelRGB8 -> Tensor
resizeAndnorm img = normalizeImage (fromRGB8ToTensor img_resized) mean std
  where
    mean = genMean (224, 224) (0.485, 0.456, 0.406)
    std = genMean (224, 224) (0.229, 0.224, 0.225)
    img_resized = centerCrop 224 224 (resizeRGB8 256 256 True img)

maxConf :: ResNet -> Image PixelRGB8 -> (Tensor, Tensor)
maxConf resnet img = (maxConf', maxConfIndex)
  where
    img' = resizeAndnorm img
    output = resnetForward resnet img'
    (maxConf', maxConfIndex) = F.maxDim (F.Dim 1) F.RemoveDim output

