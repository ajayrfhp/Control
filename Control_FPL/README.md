# Goal
- Build a control agent that chooses players to optimize FPL reward score for English premier league seasons
a
# Baseline
- Implement a random selection agent that randomly picks players within the allocated budget. 

## Design philosophy
- Is primarily just for fun
- Is secondarily a learning exercise on building data pipelines and building models that act on real world signals. 
- Meant to be an iterative effort over the years
- Always show prediction evaluation and baseline comparison
- Bonus goal
  - Be better than the random agent

## What is the type of data available ?
- We know values obtained by players in the previous seasons against various opponents



## Installation
### Enable Cuda on WSL 2 
- [Microsoft Documentation](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
  - Get Windows 11 or get the insider preview build(release preview channel) for windows 10. 
  - Get WSL 2 with kernel version 4.19.152 or higher
- [Linux environment setup on WSL](https://gist.github.com/xinzhel/6eee594f22cf6b95910dc67c40c21b94)

## Get, setup conda and install dependencies
- [Install conda on WSL](https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da)
- Create environment and install dependencies
  ```
  conda create --name test python=3.8
  conda activate test
  pip install -r requirements.txt
  ```

## Execute
- `python agent.py --run_E2E_agent == True`

### Additional links for insallation
- [Nvidia Documentation](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Setup CUDA pytorch on WSL](https://christianjmills.com/Using-PyTorch-with-CUDA-on-WSL2/)



## Resources
- [Alan turing institute](https://github.com/alan-turing-institute/AIrsenal)
- [Profiling with python](https://ymichael.com/2014/03/08/profiling-python-with-cprofile.html)
- [Awesome ML OPS](https://github.com/visenger/awesome-mlops)