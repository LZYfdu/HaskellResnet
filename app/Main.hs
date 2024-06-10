{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ParallelListComp #-}
{-# LANGUAGE BangPatterns #-}

module Main (main) where

import Codec.Picture
import ResNetFused
import LoadModel
import Classify
import System.Environment
import Torch


main :: IO ()
main = do
  args <- getArgs
  if length args < 2
    then putStrLn "Usage: stack run -- <model_path> <image_path>"   
    else do
      let model_path = return (args !! 0)
          image_path = return (args !! 1)
      model <- loadResNet50 =<< model_path
      Right img <- readImage =<< image_path
      let img' = convertRGB8 img
      
      let (probs, classes) = maxConf model img'
      let idx = asValue classes :: Int
      contents <- readFile "classes.txt"
      let linesList = lines contents
      let classifiedClass = linesList !! idx
      putStrLn $ "Predicted class: " ++ classifiedClass
      putStrLn $ "Probability: " ++ show ( probs)
      