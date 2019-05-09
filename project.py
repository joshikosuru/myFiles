import numpy as np
import math


def budgetUCB(K, B):
    mean_reward = [0.0 for i in range(K)]
    mean_cost   = [0.0 for i in range(K)]
    cost_alpha = np.random.uniform(1, 5, K)
    cost_beta = np.random.uniform(1, 5, K)
    reward_alpha = np.random.uniform(1, 5, K)
    reward_beta = np.random.uniform(1, 5, K)
    actual_mean_costs_arms = [1/(1 + cost_beta[i]/cost_alpha[i]) for i in range(K)]
    actual_mean_rewards_arms = [1/(1 + reward_beta[i]/reward_alpha[i]) for i in range(K)]
    best_arm_finder_list = [actual_mean_rewards_arms[i]/actual_mean_costs_arms[i] for i in range(K)]
    best_arm = best_arm_finder_list.index(max(best_arm_finder_list))
    lambda_ = min(actual_mean_costs_arms)
    totalCost = 0.0
    cumulativeReward = 0.0
    N = [1 for i in range(K)]
    for i in range(K):
        reward = np.random.beta(reward_alpha[i], reward_beta[i])
        cost = np.random.beta(cost_alpha[i], cost_beta[i])
        cumulativeReward += reward
        totalCost += cost
        mean_reward[i] = reward
        mean_cost[i] = cost
    T = K + 1
    while(totalCost <= B):
        D = [0.0 for i in range(K)]
        for i in range(K):
            eps = np.sqrt(2*np.log(T-1)/N[i])
            D[i] = mean_reward[i]/mean_cost[i] + eps/mean_cost[i] + (eps/mean_cost[i])*(min(mean_reward[i]+eps, 1)/max(mean_cost[i]-eps, lambda_))
        armToPull = D.index(max(D))
        reward = np.random.beta(reward_alpha[armToPull], reward_beta[armToPull])
        cost = np.random.beta(cost_alpha[armToPull], cost_beta[armToPull])
        cumulativeReward += reward
        totalCost += cost
        mean_cost[armToPull] = (mean_cost[armToPull]*N[armToPull] + cost)/(N[armToPull] + 1)
        mean_reward[armToPull] = (mean_reward[armToPull]*N[armToPull] + reward)/(N[armToPull] + 1)
        T += 1
        N[armToPull] += 1
    regret = (B/actual_mean_costs_arms[best_arm])*actual_mean_rewards_arms[best_arm] - cumulativeReward
    return regret, T

def ucb_bv1(K, B):
    mean_reward = [0.0 for i in range(K)]
    mean_cost   = [0.0 for i in range(K)]
    cost_alpha = np.random.uniform(1, 5, K)
    cost_beta = np.random.uniform(1, 5, K)
    reward_alpha = np.random.uniform(1, 5, K)
    reward_beta = np.random.uniform(1, 5, K)
    actual_mean_costs_arms = [1/(1 + cost_beta[i]/cost_alpha[i]) for i in range(K)]
    actual_mean_rewards_arms = [1/(1 + reward_beta[i]/reward_alpha[i]) for i in range(K)]
    best_arm_finder_list = [actual_mean_rewards_arms[i]/actual_mean_costs_arms[i] for i in range(K)]
    best_arm = best_arm_finder_list.index(max(best_arm_finder_list))
    lambda_ = min(actual_mean_costs_arms)
    totalCost = 0.0
    cumulativeReward = 0.0
    N = [1 for i in range(K)]
    for i in range(K):
        reward = np.random.beta(reward_alpha[i], reward_beta[i])
        cost = np.random.beta(cost_alpha[i], cost_beta[i])
        cumulativeReward += reward
        totalCost += cost
        mean_reward[i] = reward
        mean_cost[i] = cost
    T = K + 1
    while(totalCost <= B):
        D = [0.0 for i in range(K)]
        for i in range(K):
            eps = np.sqrt(2*np.log(T-1)/N[i])
            # D[i] = mean_reward[i]/mean_cost[i] + ((1 + 1/lambda_) * math.sqrt(math.log(T-1)/N[i]))/(lambda_ - math.sqrt(math.log(T-1)/N[i]))
            D[i] = mean_reward[i]/mean_cost[i] + 1.5 * ((1 + 1/lambda_) * math.sqrt(math.log(T-1)/N[i]))
        armToPull = D.index(max(D))
        reward = np.random.beta(reward_alpha[armToPull], reward_beta[armToPull])
        cost = np.random.beta(cost_alpha[armToPull], cost_beta[armToPull])
        cumulativeReward += reward
        totalCost += cost
        mean_cost[armToPull] = (mean_cost[armToPull]*N[armToPull] + cost)/(N[armToPull] + 1)
        mean_reward[armToPull] = (mean_reward[armToPull]*N[armToPull] + reward)/(N[armToPull] + 1)
        T += 1
        N[armToPull] += 1
    regret = (B/actual_mean_costs_arms[best_arm])*actual_mean_rewards_arms[best_arm] - cumulativeReward
    return regret, T

def ucb_bv2(K, B):
    mean_reward = [0.0 for i in range(K)]
    mean_cost   = [0.0 for i in range(K)]
    cost_alpha = np.random.uniform(1, 5, K)
    cost_beta = np.random.uniform(1, 5, K)
    reward_alpha = np.random.uniform(1, 5, K)
    reward_beta = np.random.uniform(1, 5, K)
    actual_mean_costs_arms = [1/(1 + cost_beta[i]/cost_alpha[i]) for i in range(K)]
    actual_mean_rewards_arms = [1/(1 + reward_beta[i]/reward_alpha[i]) for i in range(K)]
    best_arm_finder_list = [actual_mean_rewards_arms[i]/actual_mean_costs_arms[i] for i in range(K)]
    best_arm = best_arm_finder_list.index(max(best_arm_finder_list))
    lambda_ = min(actual_mean_costs_arms)
    totalCost = 0.0
    cumulativeReward = 0.0
    N = [1 for i in range(K)]
    for i in range(K):
        reward = np.random.beta(reward_alpha[i], reward_beta[i])
        cost = np.random.beta(cost_alpha[i], cost_beta[i])
        cumulativeReward += reward
        totalCost += cost
        mean_reward[i] = reward
        mean_cost[i] = cost
    T = K + 1
    while(totalCost <= B):
        D = [0.0 for i in range(K)]
        for i in range(K):
            eps = np.sqrt(2*np.log(T-1)/N[i])
            D[i] = mean_reward[i]/mean_cost[i] + (1/lambda_) * math.sqrt(math.log(T-1)/N[i]) * (1 + 1/(lambda_ - math.sqrt(math.log(T-1)/N[i])))
        armToPull = D.index(max(D))
        reward = np.random.beta(reward_alpha[armToPull], reward_beta[armToPull])
        cost = np.random.beta(cost_alpha[armToPull], cost_beta[armToPull])
        cumulativeReward += reward
        totalCost += cost
        mean_cost[armToPull] = (mean_cost[armToPull]*N[armToPull] + cost)/(N[armToPull] + 1)
        mean_reward[armToPull] = (mean_reward[armToPull]*N[armToPull] + reward)/(N[armToPull] + 1)
        T += 1
        N[armToPull] += 1
    regret = (B/actual_mean_costs_arms[best_arm])*actual_mean_rewards_arms[best_arm] - cumulativeReward
    return regret, T

def eps_greedy(K, B):
    mean_reward = [0.0 for i in range(K)]
    cost_alpha = np.random.uniform(1, 5, K)
    cost_beta = np.random.uniform(1, 5, K)
    reward_alpha = np.random.uniform(1, 5, K)
    reward_beta = np.random.uniform(1, 5, K)
    actual_mean_costs_arms = [1/(1 + cost_beta[i]/cost_alpha[i]) for i in range(K)]
    actual_mean_rewards_arms = [1/(1 + reward_beta[i]/reward_alpha[i]) for i in range(K)]
    best_arm_finder_list = [actual_mean_rewards_arms[i]/actual_mean_costs_arms[i] for i in range(K)]
    best_arm = best_arm_finder_list.index(max(best_arm_finder_list))
    best_mean_arm = actual_mean_rewards_arms.index(max(actual_mean_rewards_arms))
    delta_values = [0]*K
    for i in range(K):
        delta_values[i] = actual_mean_rewards_arms[best_mean_arm] - actual_mean_rewards_arms[i]
    delta_values[best_mean_arm] = float("inf")
    d = min(delta_values)
    totalCost = 0.0
    cumulativeReward = 0.0
    N = [0 for i in range(K)]
    T = 1
    while(totalCost <= B):
        eps = min(1, K/(T*d*d))
        if np.random.binomial(1, p = eps) == 1:
            armToPull = np.random.randint(0, K-1)
        else:
            armToPull = mean_reward.index(max(mean_reward))
        reward = np.random.beta(reward_alpha[armToPull], reward_beta[armToPull])
        cost = np.random.beta(cost_alpha[armToPull], cost_beta[armToPull])
        cumulativeReward += reward
        totalCost += cost
        mean_reward[armToPull] = (mean_reward[armToPull]*N[armToPull] + reward)/(N[armToPull] + 1)
        T += 1
        N[armToPull] += 1
    regret = (B/actual_mean_costs_arms[best_arm])*actual_mean_rewards_arms[best_arm] - cumulativeReward
    return regret, T

def UCB1(K, B):
    mean_reward = [0.0 for i in range(K)]
    cost_alpha = np.random.uniform(1, 5, K)
    cost_beta = np.random.uniform(1, 5, K)
    reward_alpha = np.random.uniform(1, 5, K)
    reward_beta = np.random.uniform(1, 5, K)
    actual_mean_costs_arms = [1/(1 + cost_beta[i]/cost_alpha[i]) for i in range(K)]
    actual_mean_rewards_arms = [1/(1 + reward_beta[i]/reward_alpha[i]) for i in range(K)]
    best_arm_finder_list = [actual_mean_rewards_arms[i]/actual_mean_costs_arms[i] for i in range(K)]
    best_arm = best_arm_finder_list.index(max(best_arm_finder_list))
    totalCost = 0.0
    cumulativeReward = 0.0
    N = [1 for i in range(K)]
    for i in range(K):
        reward = np.random.beta(reward_alpha[i], reward_beta[i])
        cost = np.random.beta(cost_alpha[i], cost_beta[i])
        cumulativeReward += reward
        totalCost += cost
        mean_reward[i] = reward
    T = K + 1
    while(totalCost <= B):
        D = [0.0 for i in range(K)]
        for i in range(K):
            D[i] = mean_reward[i] + np.sqrt(2 * np.log(T)/N[i])
        armToPull = D.index(max(D))
        reward = np.random.beta(reward_alpha[armToPull], reward_beta[armToPull])
        cost = np.random.beta(cost_alpha[armToPull], cost_beta[armToPull])
        cumulativeReward += reward
        totalCost += cost
        mean_reward[armToPull] = (mean_reward[armToPull]*N[armToPull] + reward)/(N[armToPull] + 1)
        T += 1
        N[armToPull] += 1
    regret = (B/actual_mean_costs_arms[best_arm])*actual_mean_rewards_arms[best_arm] - cumulativeReward
    return regret, T


numRuns = 100
for budg in [5, 10, 20, 50, 100]:
    totalRegret1 = 0.0
    totalRegret2 = 0.0
    # totalRegret3 = 0.0
    totalRegret4 = 0.0
    totalRegret5 = 0.0
    totalT1 = 0.0
    totalT2 = 0.0
    # totalT3 = 0.0
    totalT4 = 0.0
    totalT5 = 0.0
    for i in range(numRuns):
        np.random.seed(i+17)
        regret1, T1 = budgetUCB(100, 100*budg)
        np.random.seed(i+17)
        regret2, T2 = ucb_bv1(100, 100*budg)
        # np.random.seed(i+17)
        # regret3, T3 = ucb_bv2(100, 100*budg)
        np.random.seed(i+17)
        regret4, T4 = eps_greedy(100, 100*budg)
        np.random.seed(i+17)
        regret5, T5 = UCB1(100, 100*budg)
        totalRegret1 += regret1
        totalRegret2 += regret2
        # totalRegret3 += regret3
        totalRegret4 += regret4
        totalRegret5 += regret5
        totalT1 += T1
        totalT2 += T2
        # totalT3 += T3
        totalT4 += T4
        totalT5 += T5
    print(totalRegret1/numRuns, totalRegret2/numRuns, totalRegret4/numRuns, totalRegret5/numRuns)
    print(totalT1/numRuns, totalT2/numRuns, totalT4/numRuns, totalT5/numRuns)
