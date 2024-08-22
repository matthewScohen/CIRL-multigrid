import time

from multigrid import Type, Color, Direction
from multigrid.wrappers import FullyObsWrapper
import gymnasium as gym
import multigrid.envs
from multigrid.core.actions import Action

def load_strategy(filepath: str):
   strategy = dict()
   with open(filepath, 'r') as file:
      for line in file:
         state, action = line.strip().split(',')
         strategy[state] = Action(int(action))
   return strategy

def observation_2_state(observation, turn: int) -> str:
   grid = observation['image']
   state_string = ""
   p1_poses = list()
   p2_poses = list()
   # the grid is 8x8 but there are walls all around the edges
   play_area = grid[1:7, 1:7, :]
   for i, row in enumerate(play_area):
      for j, cell in enumerate(row):
         cell_type, color, state = cell
         if cell_type == 10: # 10 == agent
            # state is the agent orientation, multigrid uses 0 as east and increases going clockwise, the solver uses north as 0 and increases going clockwise
            state = state + 1 if state < 3 else 0
            if color == 0: # 0 == red
               p1_poses.append(f"{i}{j}{state}")
            elif color == 1: # 1 == green
               p2_poses.append(f"{i}{j}{state}")
   # construct state string
   state_string += str(turn) + "_"
   for pose in p1_poses:
      state_string += pose
   state_string += "_"
   for pose in p2_poses:
      state_string += pose

   return state_string

def set_agent_poses(env):
   env.agents[0].state.pos = (2,2)
   env.agents[0].state.dir = Direction.right

   env.agents[1].state.pos = (2, 3)
   env.agents[1].state.dir = Direction.left

def main():
   turn = 0
   num_agents = 2
   env = gym.make('MultiGrid-Empty-8x8-v0', agents=num_agents, render_mode='human')
   env = FullyObsWrapper(env)
   obs, _ = env.reset()
   actions = dict()
   # get initial observations
   set_agent_poses(env)
   for i in range(num_agents):
      actions[i] = Action.done
   observations, rewards, terminations, truncations, infos = env.step(actions)
   strategy = load_strategy("6X6_P1.strategy")
   while not env.is_done():
      for i in range(num_agents):
         if i == turn:
            if turn == 0:
               actions[i] = strategy.get(observation_2_state(observations[0], turn+1))
               # If there is no winning action then there is no value for that state in the strategy dict
               if actions[i] is None:
                  actions[i] = Action.done
               else:
                  print("winning")
            else:
               actions[i] = int(input("Enter action"))
         else:
            actions[i] = Action.done
      # actions = {agent.index: agent.action_space.sample() for agent in env.agents}
      observations, rewards, terminations, truncations, infos = env.step(actions)
      turn = (turn + 1) % num_agents

if __name__ == "__main__":
   main()