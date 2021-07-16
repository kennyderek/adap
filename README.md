# [ADAP](https://kennyderek.github.io/adap/)

### Computation Requirements
We want this project to be accessible to everyone. We use CPUs to train our models, using the framework called RLLib. In this paper, we used 3 cores and experiments ran under 24 hours.

## Set-up (<5 min)
In your favorite project directory
```
git clone https://github.com/kennyderek/adap.git
cd adap
```

Then, in a virtual environment (we recommend using conda or miniconda e.g. by ```conda create -n adapvenv python=3.8```), install the python module containing environment code (for Farmworld, Gym wrappers, etc.). This module is called ```adapenvs``` and can be installed by:
```
cd adaptation_envs
pip install -e .
```
Now, we can install the python module containing the ADAP policy code (written for RLLib) and contained in the module ```adap```. This will also install dependencies such as pytorch, tensorflow, and ray[rllib].
```
cd ..
cd adap_policies
pip install -e .
```

## Training (< 5 min)
We should be all set to run our code! Scripts are executed from the ```adap_code/scripts``` directory. To run a simple cartpole experiment using ADAP, we can follow the following steps:
```
cd ..
cd scripts
python run.py --conf ../configs/cartpole/train/adap.yaml --exp-name cartpole
```
```cartpole/adap.yaml``` is just one possible configuration file, with information regarding the 1) training environment and 2) algorithm hyperparameters. Feel free to make new configuration files by modifying hyperparameters as you wish! Automatically, RLLib will start training and checkpointing the experiment in the directory ```~ray_results/cartpole/[CONFIG_FILE + TIME]```. By default, this will checkpoint the code every 100 epochs, and at the end of training.

### Visualizing Training Results

Make sure you are using your virtual env, and that it has the installed ADAP python modules. Visualization should cause a PyGlet window to pop up, and render CartPole. 

To visualize the result, we can run 
```
python run.py --conf ../configs/cartpole/train/adap.yaml --restore ~/ray_results/cartpole/[CONFIG_FILE + TIME]/checkpoint_000025/checkpoint-25
```
for example:
```
python run.py --conf ../configs/cartpole/train/adap.yaml --restore ~/ray_results/cartpole/adap_16_12_56_54/checkpoint_000025/checkpoint-25
```
Or, we can train for longer and use a later checkpoint file.

### Evaluation on Ablations (< 1 min)
What if we want to search for ADAP policies (via latent distribution optimization) that tend to move towards the right half of our cartpole screen? Let's run the following:
```
python run.py --conf ../configs/cartpole/train/adap.yaml --restore ~/ray_results/cartpole/[CONFIG_FILE + TIME]/checkpoint_000025/checkpoint-25 --evaluate ../configs/cartpole/ablations/move_right.yaml --evolve
```
The ```--evaluate``` argument specifies a new environment configuration to use, which replaces the training environment configuration. Here, we have provided ```move_right.yaml```, which modifies the reward function to be r(t) = -x-axis position of the cartpole. The ```--evolve``` flag tells ```run.py``` to run latent optimization on the new environment dynamics.

For CartPole, we optimize the latent space for 30 steps, which is enough to recover policies from our policy space that can move left, or right, consistently. 

Awesome work! You've completed training and latent optimization of a policy space for CartPole! If you'd like, try out getting the CartPole to move on the left side of the screen, with ```move_left.yaml```.

## FAQs

### Can I run on Farmworld?
Support and instructions for Farmworld environment code will be included soon!

### Where is the ADAP code?
#### Diversity Regularizer

A lot of the code is stuff to set up experiments, visualize results, etc. The actual implementation of our diversity regularizer is located in ```adap_policies/adap/common.py```. It is implemented as a function called ```get_context_kl_loss``` (we are still updating terminology of latents versus contexts in the code base).

#### Model Types
Multiplicative and concatenation models are located in ```adap_policies/models```

