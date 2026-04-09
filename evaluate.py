from env.service_env import ServiceEnv
from agent.dqn_agent import DQNAgent
from config.config import Config

env = ServiceEnv()
state_size = len(env.reset())
action_size = env.max_tasks * env.servers

agent = DQNAgent(state_size, action_size, Config)

state = env.reset()
total_reward = 0

for _ in range(50):
    action = agent.act(state)
    state, reward, done = env.step(action)
    total_reward += reward

    if done:
        break

print("Evaluation Reward:", total_reward)