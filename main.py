import matplotlib.pyplot as plt
import numpy as np

from bandit_class import NArmedBandit

bandit = NArmedBandit(n_arms=10, mu=0, sigma=1, seed=42)

num_iter = 1_000
seed = None
softmax_avg_reward, softmax_accuracy = bandit.evaluate('softmax', num_iterations=num_iter, seed=seed)
reinforce_avg_reward, reinforce_accuracy = bandit.evaluate('reinforcement_comparison', num_iterations=num_iter, seed=seed)
gradient_avg_reward, gradient_accuracy = bandit.evaluate('gradient_bandit', num_iterations=num_iter, seed=seed)
ucb_avg_reward, ucb_accuracy = bandit.evaluate('ucb', num_iterations=num_iter, seed=seed)

print(f"Softmax: Average Reward = {softmax_avg_reward:.2f}, Accuracy = {softmax_accuracy:.2f}")
print(f"Reinforcement Comparison: Average Reward = {reinforce_avg_reward:.2f}, Accuracy = {reinforce_accuracy:.2f}")
print(f"Gradient Bandit: Average Reward = {gradient_avg_reward:.2f}, Accuracy = {gradient_accuracy:.2f}")
print(f"UCB: Average Reward = {ucb_avg_reward:.2f}, Accuracy = {ucb_accuracy:.2f}")
