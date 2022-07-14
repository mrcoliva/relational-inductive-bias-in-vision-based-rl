from common.policies import GraphNetworkPolicy
import json
import yaml
import torch
from pathlib import Path

from pprint import pprint
from common.agent import Agent

LOGFILE_NAME = 'log/run_config_log.txt'
JSON_FILEPATH = 'log/config_logs'

def log_run_config(
    agent: Agent, 
    filename: str = None,
    json_filename: str = None,
    to_stdout: bool = False) -> None:

    if filename is None:
        filename = LOGFILE_NAME
    
    if agent.model.policy == GraphNetworkPolicy:
        graph_builder = agent.model.policy.graph_builder
        graph_builder_vars = vars(graph_builder).copy()

        for k, v in graph_builder_vars.items():
            if isinstance(v, torch.Tensor):
                graph_builder_vars[k] = str(list(zip(*v.cpu().numpy())))

        agent.config['graph_builder'] = {
            'class': graph_builder.__class__.__name__,
            'vars': graph_builder_vars
        }

    config_string = yaml.dump(agent.config, default_flow_style=False)

    # append all parameters to the run log file
    with open(filename, 'a+') as f:
        components = [
            '\n\n'
            '-------------------------------------------------------',
            f'Name: {agent.name}',
            f'Logdir: {agent.log_dir}',
            f'Policy: {agent.policy if isinstance(agent.policy, str) else agent.policy.__name__}',
            '\nConfig:',
            config_string,
            'Model:',
            str(agent.model.policy),
            '-------------------------------------------------------\n\n'
        ]

        f.write('\n'.join(components))

    if to_stdout:
        print('Using config:')
        pprint(agent.config)
        print('-------------------------------------------------------')
    
    agent.config['model'] = str(agent.model.policy)

    if json_filename is None:
        json_filename = f'log/config_logs/{agent.name}.json'

    path = Path(json_filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # dump config to extra json file so we can reload it
    with open(json_filename, 'w') as file:
        file.write(json.dumps(agent.config, indent=4))