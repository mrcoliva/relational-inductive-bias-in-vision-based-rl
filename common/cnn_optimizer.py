import cv2
import torch
import torch.nn as nn

from stable_baselines3.common import logger

class EncoderNotLearningException(Exception):
    pass

class TargetPredictionOptimization(object):

    def __init__(self, config, model, net, batch_size):
        self.net = net
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()

        self.epochs_per_rollout = config['aux_noptepochs']
            
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config['aux_learning_rate'])

        self.step = 0
    
    def train(self, rollout_buffer):
        self.step += 1

        print('---------- Start optimizing CNN ---------')
        self.net._train(True)

        for epoch in range(self.epochs_per_rollout):
            running_loss = 0.0
            
            i = -1
            for rollout_data in rollout_buffer.get(self.batch_size):
                i += 1

                observations = rollout_data.observations.permute(0, 3, 1, 2).float()

                images = observations[:, 0:3, :, :]
                targets = observations[:, 3, :2, 0]

                self.optimizer.zero_grad()

                outputs, _ = self.net(images)                
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                if i % self.batch_size == self.batch_size - 1:
                    mean_loss = running_loss / self.batch_size
                    print(f'[Epoch {epoch + 1} / {self.epochs_per_rollout}] loss: {mean_loss:.6f}')
                    logger.record('cnn/loss', mean_loss)
                    running_loss = 0.0

                    if self.model.num_timesteps > 50_000 and mean_loss > 0.5:
                        raise EncoderNotLearningException(
                            f'Mean batch loss of {mean_loss:.4f} after {self.model.num_timesteps} timesteps.'
                        )

        if self.model.num_timesteps > 200_000:
            self.epochs_per_rollout = 2

        self.net._train(False)
        print('-------- Finished optimizing CNN --------')