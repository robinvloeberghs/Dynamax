# Hierarchical Bernoulli LDS

The aim of the current project is to implement a hierarchical Bernoulli Linear Dynamical System (LDS). With this model we can slow estimate drifts in decision criterion. Ultimately, this will allow to test whether slow drifts can explain several sequential effects in perceptual decision making.

The code comes from the software package Dynamax which allows the estimation of State Space Models (SSM), see https://probml.github.io/dynamax/index.html#


By Robin Vloeberghs and Scott Linderman, 2023


---

#### Installation
```

First install JAX. For Linux or macOS, the usual 'pip install jax' works. For Windows a different approach is needed (see https://github.com/google/jax/issues/5795). Files are provided by https://github.com/cloudhan/jax-windows-builder. Go to https://whls.blob.core.windows.net/unstable/index.html and download file into conda env.
More information on which file has to be chosen can be found here: https://www.reddit.com/r/learnmachinelearning/comments/qnqdy0/jax_on_windows/

conda create -n dynamax_env
conda activate dynamax_env
pip install file:C:/Users/.../Anaconda3/envs/dynamax_env/jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-win_amd64.whl
git clone https://github.com/probml/dynamax
cd dynamax
pip install -e .
```



