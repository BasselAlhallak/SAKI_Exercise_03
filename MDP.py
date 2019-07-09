import mdptoolbox
import numpy as np
import itertools


def state_generator(items_list, actions_list, number_of_positions):
    states = []
    combination_list = list(itertools.product(items_list, repeat=number_of_positions))
    for i in combination_list:
        for j in actions_list:
            b = [i, j]
            states.append(b)
    return states


def generate_probability_transition_matrix(number_of_states, probability, action):
    probability_transition_matrix = np.zeros((number_of_states, number_of_states), dtype=np.float16)
    sum_of_probabilities_in_line = probability * 256
    for row_index in range(len(states)):
        for column_index, element in enumerate(states):
            if action == element[1]:
                probability_transition_matrix[row_index][column_index] = probability / sum_of_probabilities_in_line
    return probability_transition_matrix


def generate_store_reward_matrix(states, item):
    reward_matrix = np.zeros(len(states), dtype=np.float16)
    for index, element in enumerate(states):
        items_combination = element[0]
        order = element[1]
        if "empty" == items_combination[0] and order == "store " + item:
            reward_matrix[index] = 3
        elif "empty" == items_combination[1] and order == "store " + item:
            reward_matrix[index] = 2
        elif "empty" == items_combination[2] and order == "store " + item:
            reward_matrix[index] = 1
        elif "empty" not in items_combination and order == "store " + item:
            reward_matrix[index] = -2
    return reward_matrix


def generate_restore_reward_matrix(states, item):
    reward_matrix = np.zeros(len(states), dtype=np.float16)
    for index, element in enumerate(states):
        items_combination = element[0]
        order = element[1]
        if item == items_combination[0] and order == "restore " + item:
            reward_matrix[index] = 3
        elif item == items_combination[1] and order == "restore " + item:
            reward_matrix[index] = 2
        elif item == items_combination[2] and order == "restore " + item:
            reward_matrix[index] = 1
        elif item not in items_combination and order == "restore " + item:
            reward_matrix[index] = -2
    return reward_matrix


if __name__ == "__main__":

    items_list = ["empty", "red", "blue", "white"]
    actions_list = ["store red", "store blue", "store white", "restore red", "restore blue", "restore white"]

    states = state_generator(items_list, actions_list, 4)

    store_red_matrix = generate_probability_transition_matrix(len(states), 0.246, "store red")
    store_blue_matrix = generate_probability_transition_matrix(len(states), 0.123, "store blue")
    store_white_matrix = generate_probability_transition_matrix(len(states), 0.152, "store white")
    restore_red_matrix = generate_probability_transition_matrix(len(states), 0.230, "restore red")
    restore_blue_matrix = generate_probability_transition_matrix(len(states), 0.123, "restore blue")
    restore_white_matrix = generate_probability_transition_matrix(len(states), 0.123, "restore white")

    store_red_reward = generate_store_reward_matrix(states, "red")
    store_blue_reward = generate_store_reward_matrix(states, "blue")
    store_white_reward = generate_store_reward_matrix(states, "white")
    restore_red_rewards = generate_restore_reward_matrix(states, "red")
    restore_blue_rewards = generate_restore_reward_matrix(states, "blue")
    restore_white_rewards = generate_restore_reward_matrix(states, "white")

    tmp_full = np.array([store_red_matrix, store_blue_matrix, store_white_matrix,
                         restore_red_matrix, restore_blue_matrix, restore_white_matrix])

    reward_full = np.array([store_red_reward, store_blue_reward, store_white_reward,
                           restore_red_rewards, restore_blue_rewards, restore_white_rewards]).T

    mdp_result_policy = mdptoolbox.mdp.PolicyIteration(tmp_full, reward_full, 0.3, max_iter=100)
    mdp_result_value = mdptoolbox.mdp.ValueIteration(tmp_full, reward_full, 0.3, max_iter=100)

    mdp_result_policy.run()
    mdp_result_value.run()

    print('PolicyIteration:')
    print(mdp_result_policy.policy)
    print(mdp_result_policy.V)
    print(mdp_result_policy.iter)

    print('ValueIteration:')
    print(mdp_result_value.policy)
    print(mdp_result_value.V)
    print(mdp_result_value.iter)





