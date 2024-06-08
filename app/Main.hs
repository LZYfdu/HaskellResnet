{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import Torch
import LoadModel

main :: IO ()
main = do
    let t = ones' [2, 2]
    print t
