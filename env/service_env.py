import numpy as np
import random

class ServiceEnv:
    def __init__(self, max_tasks=5, servers=2):
        self.max_tasks = max_tasks
        self.servers = servers
        self.reset()

    def reset(self):
        self.queue = [random.randint(1, 5) for _ in range(self.max_tasks)]
        self.servers_free = [1] * self.servers
        return self._get_state()

    def _get_state(self):
        return np.array(self.queue + self.servers_free, dtype=np.float32)

    def step(self, action):
        task_idx = action // self.servers
        server_idx = action % self.servers

        reward = 0

        if task_idx < len(self.queue) and self.servers_free[server_idx] == 1:
            task_time = self.queue[task_idx]

            reward += 10 - task_time
            self.servers_free[server_idx] = 0
            self.queue[task_idx] = 0
        else:
            reward -= 5

        reward -= sum([1 for t in self.queue if t > 0])

        # random server freeing
        for i in range(self.servers):
            if random.random() < 0.3:
                self.servers_free[i] = 1

        done = all(t == 0 for t in self.queue)
        return self._get_state(), reward, done