import numpy as np

def calculate_mean_reward(model, env, n_runs=1, seeds = None, render: bool = False):
    """
    A function to calculate a mean reward given a model and an environment to test on. It can
    handle environments that have variable number of timesteps (e.g. if terminal condition is reached before max steps)
    :param model: the model to use for prediction
    :param env: the environment to test on (it is expected that VecEnv environment type is provided)
    :param n_runs: how many runs to perform (e.g. usually the VecEnv has X processes where X is number of CPUs), so for
    for more episodes n_runs is used such that n_runs*X episodes will be executed
    :return: a mean reward, calculated as the average of all episode rewards
    :param render: Whether to render the environment during evaluation.
    """
    episode_rewards = []
    total_steps = 0
    
    for _ in range(n_runs):
        if seeds is not None:
            try:
                env.seed(seeds)
            except:
                pass

        obs = env.reset()
        cum_reward = 0
       
        # running_envs is a mask to make each environment in each process run only once in cases of different number
        # of possible timesteps per environment (usually due to early environment solving due to terminal condition
        # other than the maximum number of timesteps). once all environments have completed the run, each environment is
        # considered again
        running_envs = np.ones(env.num_envs, dtype=bool)
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # set the actions to 0 for finished envs (0 usually interpreted as a "do nothing" action)
            action = action * running_envs[:, None]
            obs, rewards, dones, _ = env.step(action)

            if render:
                try:
                    env.render()
                except:
                    pass
        
            # use the reward per timestep only from the environments that are still running
            cum_reward += (rewards * running_envs)
            total_steps += np.sum(running_envs)
            
            # update the running envs (sets to 0 the ones that had terminated in this timestep)
            running_envs = np.multiply(running_envs, np.bitwise_not(dones))
        
            if not np.any(running_envs):
                episode_rewards.append(cum_reward)
                break

    return np.array(episode_rewards).reshape(-1), total_steps