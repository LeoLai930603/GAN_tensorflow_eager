# GAN models with TensorFlow-eager implementation

## Introduction

TensorFlow 2.0 is coming soon, but I found most GAN demos are still working on the static graphs.  
Therefore, this repo is for a `TensorFlow-eager` (which is default enabled in TF 2.0) implementation of those well-known demos.  
Most demo examples are from: [wiseodd/generative-models](https://github.com/wiseodd/generative-models). The author has already implemented those models in Pytorch and TensorFlow.

## Dependency

```TensorFlow==1.13.0```  
```matplotlib>=3.0.2```

## Usage

Feel free to explore the files in this project, the model implementation is in the directory with its name. For example, the vanilla GAN is implemented in `vanilla_gan/gan_eager.py`.  
Only entry of each demo is the `main()` function, so the demo procedure is:  
1. Find out the script with ``main()`` function. (Don't be afraid with the numerous files in each folder, I will try to minimize it to just a few files)
2. Assume you have found the expected file: `foo/bar.py`, then open the shell in the project directory, input ```python foo/bar.py```.
3. All set! You could observe the training process (with logs on screen) and the outputs of GAN during training.

## Change Log
### v0.1.0(2019/06/13 11:30 UTC+08:00)
* Basic project setup (only vanilla GAN is available at this time)
* README.md added
### v0.1.1(2019/06/23 15:15 UTC+08:00)
* Conditional GAN project added