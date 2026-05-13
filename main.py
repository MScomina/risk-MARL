from environment import risk
from pettingzoo.test import api_test, performance_benchmark
from gymnasium.spaces.utils import flatten, unflatten, flatdim

env = risk.env(render_mode="human", n_agents=3, max_armies=300)
api_test(env, num_cycles=1000, verbose_progress=True)
performance_benchmark(env)