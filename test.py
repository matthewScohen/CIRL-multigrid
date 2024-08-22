import time
import gymnasium as gym
import multigrid.envs
from multigrid.core.actions import Action

turn = 0
num_agents = 4
env = gym.make('MultiGrid-Empty-8x8-v0', agents=num_agents, render_mode='human', fully_obs=True, highlight=False)
observations, infos = env.reset()

while not env.is_done():
   # this is where you would insert your policy / policies
   actions = dict()
   for i in range(num_agents):
      if i == turn:
         actions[i] = env.agents[turn].action_space.sample()
      else:
         actions[i] = Action.done
   # actions = {agent.index: agent.action_space.sample() for agent in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)
   turn = (turn + 1) % num_agents
   # time.sleep(0.5)

env.close()