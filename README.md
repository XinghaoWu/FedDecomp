# Introduction

This is the implementation of our paper "Decoupling General and Personalized Knowledge in Federated Learning via Additive and Low-rank Decomposition."

# Dataset

Our experiments utilize three datasets: CIFAR-10, CIFAR-100, and Tiny ImageNet. CIFAR-10 and CIFAR-100 will automatically download when the code is run. For the Tiny ImageNet dataset, one can download the dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract it to the `./data/` directory.

# System

- `main.py`: Entry point of the program.
- `./utils/options.py`: Configuration of experimental hyperparameters.
- `./src/client.py`: Client-side code.
- `./src/server.py`: Server-side code.
- `./models`: Directory for storing backbone model code.

# Simulation

## Environment

The required experimental environments can be found in `requirements.txt`.

## Training and Evaluation Demo

- Experiment with CIFAR-10 dataset.

  ```shell
  python main.py --num_users=40 --dataset=cifar --model=resnet8 --alpha=1.0 --Conv_r=0.6 --Linear_r=4 --local_p_ep=2
  ```


Experimental results can be found in the `./log` directory.

