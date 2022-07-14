# Graph Neural Networks for Relational Inductive Bias in Vision-based Deep Reinforcement Learning of Robot Control

This repository contains the code to reproduce the results of the paper [Graph Neural Networks for Relational Inductive Bias in Vision-based Deep Reinforcement Learning of Robot Control](https://arxiv.org/abs/2203.05985) by Marco Oliva, Soubarna Banik, Josip Josifovski and Alois Knoll.

<img src="6link_control.gif" width="700">

# 🔨 Installation
All of the code and the required dependencies are packaged in a docker image. To install, follow these steps:

* if you run it for the first time, clone the repo `cd` into it.
* build the Docker image with: `docker build -t cuda_geometric .`
* run the image with: 
    ```bash
    docker run -it --rm --gpus all -v $(pwd):/code --user $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro --network=host cuda_geometric
    ```

From within the container, you can train a new agent or load and evaluate an already trained agent.

## 🗂 Models

The models are defined by the following identifier which denotes the environment in which it acts and the model architecture.
Models with the prefix `2_link` use the `nlinks_box2d` environment, whereas the prefix `6_link` denotes models using the KUKA robot in the Acrobot Unity simulation.

* `2_link_gn`
* `2_link_mlp`
* `2_link_cnn_mlp`
* `2_link_cnn_gn`
* `2_link_cnn_mlp_img`
* `6_link_gn`
* `6_link_mlp`
* `6_link_cnn_mlp`
* `6_link_cnn_gn`
* `6_link_cnn_mlp_img`
* `6_link_gn_local_controllers`
* (...and various others used in ablation studies)

> **__NOTE__** When using one of the '6_link...' models, the acrobot unity simulation must be started first.

## 🤖 How to run the acrobot simulation containing the 6-DoF robot.
It is recommended to launch the unity simulation from _outside_ the Docker image.

Follow the following steps to start it from the root directory of the repository (note the display id might be different depending on `$DISPLAY`):
* navigate into the env's directory: `cd envs/unity_manipulator/v0.7/`
* check the `configuration.xml` file: if you'd like to train/evaluate a model that is based on image observations (contains 'cnn' in the identifier), `EnableObservationImage` must be `true`, otherwise it should be set to `false`.
* run the executable: `DISPLAY=:1 ./ManipulatorEnvironment.x86_64`
* if using ROS as communication mode, start a rose node with ```docker run --rm -it --name "ros_bridge" --network=host --gpus all ctonic/ros-bridge```

## 🏋️ Train an agent

To reproduce the results obtained in the thesis, the `run_experiments.py` script allows to train all of the various available models. To select which model to train, please refer to the documentation within the script.

If selecting one of the 6-link models for training in the script, remember to start the simulation first (see above).
```python
xvfb-run -s "-screen 0 200x200x24" python run_experiments.py
```

This will start training the agent for one million time steps, writing TensorBoard log data to the `/tb_logs` directory, and saving the model in the `/models` directory when completed. It will also save a new `json` file to `/log/config_logs` which contains the used configuration, including hyperparameters and model definitions.

## 📈 Watch the model train in TensorBoard:
To watch the progress statistics of the agent during (or after) training, simply launch a TensorBoard session and pass the `/tb_logs` directory as the `--logdir` argument.

```bash
tensorboard --logdir=tb_logs
```

# Perform hyperparameter search
We provide a script to run hyperparameter optimization for the `GN` and `CNN-GN` models.
Just execute the following and provide as argument either `-a GN` or `-a CNN-GN`, e.g.
```python
xvfb-run -s "-screen 0 200x200x24" python hpo.py -a CNN-GN
```

You can customize the parameters you want to optimize over, as well as their possible values and sample strategies in `/hpo/utils.py`.
