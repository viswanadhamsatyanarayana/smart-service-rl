from env.service_env import ServiceEnv
from agent.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer
from config.config import Config

import numpy as np
import torch
import matplotlib.pyplot as plt

env = ServiceEnv()
state_size = len(env.reset())
action_size = env.max_tasks * env.servers

agent = DQNAgent(state_size, action_size, Config)
memory = ReplayBuffer(Config.MEMORY_SIZE)

rewards_history = []

for episode in range(Config.EPISODES):
    state = env.reset()
    total_reward = 0
    waiting_time = 0

    for step in range(Config.MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # waiting time = number of pending tasks
        waiting_time += sum([1 for t in next_state[:env.max_tasks] if t > 0])

        memory.push(state, action, reward, next_state, done)
        agent.train(memory, Config.BATCH_SIZE)

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_history.append(total_reward)

    if episode % Config.TARGET_UPDATE == 0:
        agent.update_target()

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Waiting: {waiting_time}")

# ✅ Save model
torch.save(agent.model.state_dict(), "dqn_model.pth")

# ✅ Plot training curve
plt.plot(rewards_history)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Training Performance")
plt.savefig("training_plot.png")

print("Training complete. Model saved!")