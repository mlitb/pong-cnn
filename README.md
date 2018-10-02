# Pong cnn
> Pong is one of the earliest arcade video games. It is a table tennis sports game featuring simple two-dimensional graphics. The game was originally manufactured by Atari, which released it in 1972. Allan Alcorn created Pong as a training exercise assigned to him by Atari co-founder Nolan Bushnell.

### Introduction
This is the implementation of training a RL agent to play the mighty game of Pong.

### Requirements
In order to train a RL agent to play the mighty game of Pong, you need to run it on **Python3** and install:
1. gym
```
pip install gym
```
2. gym atari
```
pip install "gym[atari]"
```
3. numpy
```
pip install numpy
```

### Train RL agent
Run this command on your terminal
```
python pong.py
```
```
Arguments :
-l, --load <pkl_path> #path to the saved model to load from
-s, --save <folder_path> #path to the folder to save model
-r, --render #whether to render the environment or not
```
### Authors
1. Faza Fahleraz https://github.com/ffahleraz
2. Nicholas Rianto Putra https://github.com/nicholaz99
3. Abram Perdanaputra https://github.com/abrampers

### Words from Authors
Thanks to Andrej Karpathy for his amazing blogpost about [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/).
> *"It shouldn’t work, but amusingly we live in a universe where it does" - Andrej Karpathy*

### References
* [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
* [220 Logistic Regression](https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf)
* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2)
* [Lecture 14 | Deep Reinforcement Learning](https://www.youtube.com/watch?v=lvoHnicueoE&t=464s)
* [TensorFlow and deep reinforcement learning, without a PhD (Google I/O '18)](https://www.youtube.com/watch?v=t1A3NTttvBA&t=873s)
* [Building a Convolutional Neural Network in Python with Tensorflow](https://medium.com/data-science-group-iitr/building-a-convolutional-neural-network-in-python-with-tensorflow-d251c3ca8117)
