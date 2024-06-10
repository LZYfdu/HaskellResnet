
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


{-
type Img = Tensor

getMinl :: 
  -- | input image
  Img -> 
  -- | minimum size of the face
  Float -> 
  -- | minl
  Float
getMinl img minsize =  minl 
  where
    minl = (fromIntegral minl_i) * m
    m = 12.0 / minsize
    minl_i = Prelude.min  ((shape img) !! 2) ((shape img) !! 3 )


createScalePyramid :: 
  -- | scale_i, init: 12.0/minsize
  Float -> 
  -- | factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
  Float -> 
  -- | minl
  Float ->
  -- | scales
  [Float]
createScalePyramid scale_i factor minl
  | minl >= 12.0 = [scale_i] ++ createScalePyramid (scale_i * factor) factor (minl * factor)
  | otherwise = []

tripleAppend :: ([a], [b], [c]) -> ([a], [b], [c]) -> ([a], [b], [c])
tripleAppend (a1, b1, c1) (a2, b2, c2) = (a1 ++ a2, b1 ++ b2, c1 ++ c2)

generateBoundingBoxes :: 
  -- | reg
  Tensor -> 
  -- | probs
  Tensor -> 
  -- | scale
  Float -> 
  -- |threshold
  Float ->
  -- | bounding boxe
  (Tensor,
  -- | image_ind
   Tensor)
generateBoundingBoxes reg probs scale threshold = (box, image_inds)
  where
    stride = 2.0
    cellsize = 12.0
    reg' = F.permute [1, 0, 2, 3] reg
    mask = (probs >= threshold)
    mask_inds = F.nonzero mask
    image_inds = select 1 1 mask_inds
    score = F.maskedSelect mask probs
    reg'' = F.permute [1, 0] ( F.view [shape reg' !! 0, (shape reg' !! 2) * (shape reg' !! 3)] (F.maskedSelect mask reg'))
    bb = FI.flip (toType Float (indexSelect' 1 [1..] mask_inds)) [1]
    q1 = F.floor (F.divScalar scale (F.addScalar 1.0 (F.mulScalar stride bb)))
    q2 = F.floor (F.divScalar scale (F.addScalar cellsize (F.mulScalar stride bb)))
    box = F.cat [1] [q1, q2, F.unsqueeze 1 score, reg'']

firstStage :: 
  -- | model
  PNetBB ->
  -- | input image
  Image PixelRGB8 ->
  -- | offset
  Float ->
  -- | threshold[0]
  Float ->
  -- | scales
  [Float] ->
  -- | bounding boxes
  ([Tensor],
  -- | image_inds
  [Tensor],
  -- | scale_picks
  [Float])
firstStage pnet img offset threshold [] = ([], [], [])
firstStage pnet img offset threshold (scale:scales) = tripleAppend ([box], [image_ind], [scale_pick]) (firstStage pnet img offset_new threshold scales)
  where
    im_data = downScaleImage img (floor (scale * (fromIntegral (imageWidth img))) + 1) (floor (scale * (fromIntegral (imageHeight img))) + 1)
    im_data' = fromRGB8ToTensor im_data
    im_data'' = mulScalar 0.0078125 (subScalar 127.5 im_data')
    (reg, probs) = pnetBBForward pnet im_data''
    (box, image_ind) = generateBoundingBoxes reg probs scale threshold 
    pick = 
-}