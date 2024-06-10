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
import System.CPUTime


main :: IO ()
main = do
  args <- getArgs
  if length args < 2
    then putStrLn "Usage: stack run -- <model_path> <image_path>"   
    else do
      
      time1 <- getCPUTime

      let model_path = return (args !! 0)
          image_path = return (args !! 1)
      model <- loadResNet50 =<< model_path
      Right img <- readImage =<< image_path
      let img' = convertRGB8 img
      
      time2 <- getCPUTime
      let elapsed1 = fromIntegral (time2 - time1) / (10^12)

      putStrLn $ "Time elapsed for loading model and image: " ++ show elapsed1 ++ " seconds"

      start <- getCPUTime

      let (probs, classes) = maxConf model img'

      end <- getCPUTime
      let elapsed = fromIntegral (end - start) / (10^12)
      putStrLn $ "Time elapsed for preprocessing and predicting: " ++ show elapsed ++ " seconds"

      time3 <- getCPUTime
      let idx = asValue classes :: Int
      contents <- readFile "classes.txt"
      let linesList = lines contents
      let classifiedClass = linesList !! idx
      time4 <- getCPUTime
      let elapsed2 = fromIntegral (time4 - time3) / (10^12)
      putStrLn $ "Time elapsed for classifying: " ++ show elapsed2 ++ " seconds"
      putStrLn $ "Predicted class: " ++ classifiedClass
      putStrLn $ "Probability: " ++ show ( probs)
      