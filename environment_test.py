from environment import risk
from pettingzoo.test import api_test, performance_benchmark
import time
import numpy as np

N_PLAYERS = 4
RENDER_COOLDOWN = 0.05

RUN_STABILITY = True
RUN_RENDER = True

def stability_benchmarks():
    env = risk.env(render_mode=None, n_agents=N_PLAYERS)
    api_test(env, num_cycles=3000, verbose_progress=True)
    performance_benchmark(env)

def render_benchmarks():
    env = risk.env(render_mode="human", n_agents=N_PLAYERS)
    env.reset()
    done_agents = {agent: False for agent in env.agents}

    while not all(done_agents.values()):
        done_agents = np.logical_or(env.terminations, env.truncations)
        current_agent = env.agent_selection
        if done_agents[current_agent]:
            continue

        mask = env.observe(current_agent)["action_mask"]
        act = env.action_space(current_agent).sample(mask=mask)
        env.step(act)

        # Render the updated state
        env.render()

        # Small delay so you can see something happen
        time.sleep(RENDER_COOLDOWN)

    env.close()

def main():
    if RUN_STABILITY:
        stability_benchmarks()
    if RUN_RENDER:
        render_benchmarks()

if __name__ == "__main__":
    main()