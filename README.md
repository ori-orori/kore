# kore
## requirements

### conda virtual environmet setup

```python
$ conda create -n "environment name" python=3.8
$ conda activate "environment name"
```

### Install dependencies

```python
$ pip install numpy matplotlib
$ pip install gym
$ pip install kaggle-environments
$ pip install tqdm
```

### Install pytorch

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Environments

about kore

## Create data

You can use the collected data or create new data. 

### Use collected data

In order to use the collected data, the zip file must be decompressed.

```python
$ cd dataset
$ python make_data.py --mode unzip --path ./data.zip
```

The decompressed data has the following structure.

```python
...
data
├── beta_1st
│   ├── beta_1st
│   │   ├── 000000.json
│   │   ├── 000001.json
│   │   ├── 000002.json
│   │   ├── ...
│   ├── beta_6th
│   │   ├── 000000.json
│   │   ├── 000001.json
│   │   ├── 000002.json
│   │   ├── ...
│   ├── opponent
│   │   ├── 000000.json
│   │   ├── 000001.json
│   │   ├── 000002.json
│   │   ├── ...
```

### Create new data

You have to modify config/data.yaml file before create new data.

```python
# config/data_config.yaml
agent: beta_1st # agent you want to collect
other_agents: [beta_1st, beta_6th, opponent] # opponent agents
samples_num: 1000 # number of data to collect per other agent
```

After modifying the config file, then type the command below. ****It takes about a minute to collect one data.

```python
$ cd dataset
$ python make_data.py --mode make --config ../config/data_config.yaml
```

The created data has the following structure.

```python
...
data
├── agent
│   ├── other agent1
│   │   ├── 000000.json
│   │   ├── 000001.json
│   │   ├── 000002.json
│   │   ├── ...
│   ├── other agent2
│   │   ├── 000000.json
│   │   ├── 000001.json
│   │   ├── 000002.json
│   │   ├── ...
```

## Train

Training consists of two stages. The first is supervised learning, and the second is reinforcement learning. You can modify the config file before training.

### Supervised learning

```python
$ python train.py --config ./config/train_config.yaml --mode sl
```

### Reinforcement learning

```python
$ python train.py --config ./config/train_config.yaml --mode rl
```

## Test
