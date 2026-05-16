from environment import risk
from pettingzoo.test import api_test, performance_benchmark

env = risk.env(render_mode="human", n_agents=3, max_armies=100)
api_test(env, num_cycles=10000, verbose_progress=True)
performance_benchmark(env)