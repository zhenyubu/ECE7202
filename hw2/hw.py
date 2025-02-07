#By: Zhenyu Bu, 2025-02-06

import numpy as np
state_to_pos = {
    0:  (0, 0),
    1:  (0, 1),
    2:  (0, 2),
    3:  (1, 0),
    4:  (1, 2),
    5:  (2, 0),
    6:  (2, 1),
    7:  (2, 2),
    8:  (3, 0),
    9:  (3, 1),   # trap
    10: (3, 2),   # goal
    11: (1, 1),   # block
}
pos_to_state = {pos: s for s, pos in state_to_pos.items()}
print(f"state_to_pos: {state_to_pos}")
TERMINAL = 12

def get_next_state_and_reward(s, action):

    if s == TERMINAL:
        return (TERMINAL, 0.0)

    # Current position in the grid
    if s in state_to_pos:
        row, col = state_to_pos[s]
    else:
        return (TERMINAL, 0.0)  # Just in case

    # Proposed move
    if action == 'U':
        new_row, new_col = row + 1, col
    elif action == 'D':
        new_row, new_col = row - 1, col
    elif action == 'L':
        new_row, new_col = row, col - 1
    elif action == 'R':
        new_row, new_col = row, col + 1
    else:
        new_row, new_col = row, col

    if not (0 <= new_row <= 3 and 0 <= new_col <= 2):
        new_row, new_col = row, col  # bounce

    if (new_row, new_col) == (1, 1):
        new_row, new_col = row, col  # bounce
    ns = pos_to_state[(new_row, new_col)]

    if ns == 9:
        return (TERMINAL, -10)
    if ns == 10:
        return (TERMINAL, 10)

    return (ns, -1)


all_states = list(range(13))
# updatable_states = [s for s in all_states][::-1]
updatable_states = [s for s in all_states]

# shuffle = np.random.permutation(updatable_states)
# updatable_states = shuffle


V = {s: 0.0 for s in all_states}
gamma = 1
theta = 1e-8
actions = ['U','D','L','R']

iteration = 0

while True:
    delta = 0.0
    newV = dict(V)
    iteration += 1

    for s in updatable_states:
        val = 0.0
        for a in actions:
            if s == 9:
                ns, r = get_next_state_and_reward(s, a)
                val += 0.25 * (r + gamma * (-10))
            elif s == 10:
                ns, r = get_next_state_and_reward(s, a)
                val += 0.25 * (r + gamma * (10))
            else:
                ns, r = get_next_state_and_reward(s, a)
                val += 0.25 * (r + gamma * V[ns])
        # Update
        delta = max(delta, abs(val - newV[s]))
        newV[s] = val

    V = newV

    if iteration % 95 == 0:
        print("\nGrid visualization (values) %d:" % iteration)
        print("-" * 42)
        for col in range(2, -1, -1):  # Start from top row (visual)
            line = "|"
            for row in range(4):  # Go through columns
                if (row, col) in pos_to_state:
                    s = pos_to_state[(row, col)]
                    if s == 11:  # block
                        line += "  Block  |"
                    else:
                        if V[s] > 0:
                            line += f" {V[s]:+6.3f} |"
                        else:
                            line += f" {V[s]:6.3f} |"
                        if s == 10:  # goal
                            line = line[:-1] + "(G)|"
                        elif s == 9:  # trap
                            line = line[:-1] + "(T)|"
                else:
                    line += "         |"
            print(line)
            print("-" * 42)


    if delta < theta:
        break

# Asynchronous update
# while True:
#     iteration += 1
#     delta = 0.0


#     ordering = updatable_states  # default

#     # Asynchronous update: update V in place.
#     for s in ordering:
#         v_old = V[s]
#         v_new = 0.0
#         for a in actions:
#             if s == 9:
#                 ns, r = get_next_state_and_reward(s, a)
#                 v_new += 0.25 * (r + gamma * (-10))
#             elif s == 10:
#                 ns, r = get_next_state_and_reward(s, a)
#                 v_new += 0.25 * (r + gamma * (10))
#             else:
#                 ns, r = get_next_state_and_reward(s, a)
#                 v_new += 0.25 * (r + gamma * V[ns])
#         V[s] = v_new
#         delta = max(delta, abs(v_new - v_old))

#     if delta < theta:
#         break


print(f"\nConverged in {iteration} iterations.")
# for s in range(13):
#     print(f"V({s}) = {V[s]:.3f}")

print("\nGrid visualization (values) Final:")
print("-" * 42)
for col in range(2, -1, -1):  
    line = "|"
    for row in range(4):  # Go through columns
        if (row, col) in pos_to_state:
            s = pos_to_state[(row, col)]
            if s == 11:  # block
                line += "  Block  |"
            else:
                if V[s] > 0:
                    line += f" {V[s]:+6.3f} |"
                else:
                    line += f" {V[s]:6.3f} |"
                if s == 10:  # goal
                    line = line[:-1] + "(G)|"
                elif s == 9:  # trap
                    line = line[:-1] + "(T)|"
        else:
            line += "         |"
    print(line)
    print("-" * 42)
