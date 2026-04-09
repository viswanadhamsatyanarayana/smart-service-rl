import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from env.service_env import ServiceEnv
from agent.dqn_agent import DQNAgent
from config.config import Config

# ----------------------------
# Setup
# ----------------------------
st.title("🚀 Smart Service Management (RL vs FIFO)")

env = ServiceEnv()
state_size = len(env.reset())
action_size = env.max_tasks * env.servers

agent = DQNAgent(state_size, action_size, Config)

# ✅ Load trained model
try:
    agent.model.load_state_dict(torch.load("dqn_model.pth"))
    agent.epsilon = 0  # no exploration
    st.success("Loaded trained RL model")
except:
    st.warning("No trained model found. Train first!")

# ----------------------------
# Session State
# ----------------------------
if "state" not in st.session_state:
    st.session_state.state = env.reset()
    st.session_state.total_reward = 0
    st.session_state.steps = 0
    st.session_state.history = []
    st.session_state.waiting_time = 0

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")

mode = st.sidebar.selectbox("Mode", ["RL Agent", "FIFO"])

step_btn = st.sidebar.button("▶ Step")
reset_btn = st.sidebar.button("🔄 Reset")
auto_btn = st.sidebar.button("⚡ Auto Run (50 steps)")

# ----------------------------
# Reset
# ----------------------------
if reset_btn:
    st.session_state.state = env.reset()
    st.session_state.total_reward = 0
    st.session_state.steps = 0
    st.session_state.history = []
    st.session_state.waiting_time = 0

# ----------------------------
# FIFO Policy
# ----------------------------
def fifo_policy(state):
    queue = state[:env.max_tasks]
    servers = state[env.max_tasks:]

    for i, task in enumerate(queue):
        if task > 0:
            for j, s in enumerate(servers):
                if s == 1:
                    return i * env.servers + j
    return 0

# ----------------------------
# Step Function
# ----------------------------
def run_step():
    state = st.session_state.state

    if mode == "RL Agent":
        action = agent.act(state)
    else:
        action = fifo_policy(state)

    next_state, reward, done = env.step(action)

    st.session_state.state = next_state
    st.session_state.total_reward += reward
    st.session_state.steps += 1
    st.session_state.history.append(st.session_state.total_reward)

    # waiting time
    st.session_state.waiting_time += sum(
        [1 for t in next_state[:env.max_tasks] if t > 0]
    )

    return done

# ----------------------------
# Run Step / Auto
# ----------------------------
if step_btn:
    run_step()

if auto_btn:
    for _ in range(50):
        if run_step():
            break

# ----------------------------
# Display Queue
# ----------------------------
state = st.session_state.state
queue = state[:env.max_tasks]
servers = state[env.max_tasks:]

st.subheader("📋 Task Queue")
st.bar_chart(queue)

st.subheader("🖥 Server Status")
st.bar_chart(servers)

# ----------------------------
# Metrics
# ----------------------------
st.subheader("📊 Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Steps", st.session_state.steps)
col2.metric("Total Reward", round(st.session_state.total_reward, 2))
col3.metric("Waiting Time", st.session_state.waiting_time)

# ----------------------------
# Reward Graph
# ----------------------------
st.subheader("📈 Reward Over Time")

if len(st.session_state.history) > 1:
    fig, ax = plt.subplots()
    ax.plot(st.session_state.history)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")

    st.pyplot(fig)