import numpy as np
from model import NeuralNetwork, softmax

import multiprocessing
import threading
import gym
from time import sleep
from gym import spaces, wrappers


class GymSimulation():

    def __init__(self,
                 layer_sizes,
                 env_name,
                 invert_reward=True,
                 amount_threads=None,
                 cpu_only=True,
                 number_repeats=5,
                 seed=8,
                 reward_offset=0):
        self.reward_offset = reward_offset
        self.invert_reward = invert_reward
        self.amount_threads = multiprocessing.cpu_count() if amount_threads == None else amount_threads
        self.envs = [gym.make(env_name) for _ in range(self.amount_threads)]
        for env in self.envs:
            env.seed(seed)
        self.env_name = env_name
        input_size = self.envs[0].observation_space.shape[0]

        if isinstance(self.envs[0].action_space, spaces.Discrete):
            print(env_name, "has a discrete action space")
            output_activation = softmax
            output_size = self.envs[0].action_space.n
            self.discrete_action = True
            output_activation = softmax

        else:
            print(env_name, "has a non-discrete action space")
            output_activation = None
            output_size = self.envs[0].action_space.shape[0]
            self.discrete_action = False
            output_activation = np.tanh

        self.models = [NeuralNetwork(
            input_size=input_size,
            hidden_sizes=layer_sizes,
            output_size=output_size,
            output_activation=output_activation,
            discrete_action=self.discrete_action
        ) for _ in range(self.amount_threads)]

        self.solution_size = len(self.models[0].get_weights())
        self.number_repeats = number_repeats
        self.interrupt = False


    def get_solution_size(self):
        return len(self.models[0].get_weights())


    def preview_weights(self, weights, timesteps=200):

        model = self.models[0]

        env = gym.make(self.env_name)
        env = wrappers.Monitor(env, ".", force=True)
        env.reset()
        env.render(close=True)

        observation = env.reset()

        for i in range(timesteps):

            env.render()

            action = model.forward(observation)
            observation, reward, done, info = env.step(action)

            if done == True:
                print("Done early at step", i)
                break
            sleep(1.0/60.0)
        env.close()


    def get_fitnesses(self, population):
        population_size = len(population)

        cpu_amount = self.amount_threads
        if cpu_amount > 1:
            amount_per_thread = int(np.floor(population_size / cpu_amount))
            left_over = population_size - amount_per_thread * cpu_amount
        else:
            amount_per_thread = len(population)
            left_over = 0

        fitnesses = np.zeros(population_size)

        def fill_fitnesses(begin, size, env, model):

            for i in range(begin, begin + size):

                weights = population[i]
                model.set_weights(weights)

                total_reward = 0
                total_steps = 0
                for _ in range(self.number_repeats):
                    observation = env.reset()
                    local_reward = 0
                    local_steps = 0
                    for j in range(100000):

                        action = model.forward(observation)
                        observation, reward, done, info = env.step(action)
                        local_reward += reward
                        local_steps = j
                        if done or self.interrupt:
                            break
                    total_steps += local_steps
                    total_reward += local_reward

                    if self.interrupt:
                        break

                fitnesses[i] = total_reward / self.number_repeats

                if self.invert_reward:
                    fitnesses[i] *= -1

                if self.reward_offset != 0:
                    fitnesses[i] += self.reward_offset

                if self.interrupt:
                    break

        threads = []

        try:
            index = 0
            for i in range(cpu_amount):
                population_amount = (amount_per_thread + 1) if i < left_over else amount_per_thread
                arguments = [index, population_amount, self.envs[i], self.models[i]]
                thread = threading.Thread(target=fill_fitnesses,
                                          args=arguments)
                threads.append(thread)
                index += population_amount

            for t in threads:
                t.start()
            for t in threads:
                t.join()
        except (KeyboardInterrupt, SystemExit):
            self.interrupt = True
            for t in threads:
                t.join()
            print('Exiting')
            self.interrupt = False
            raise
        except Exception as error:
            print('Caught this error:\n' + repr(error))



        return fitnesses
