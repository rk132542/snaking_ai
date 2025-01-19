# import pygame
import logging
import numpy as np

import tensorflow as tf

import numpy as np
import random
from collections import deque

# WIDTH, HEIGHT = 650, 650
# FPS = 1

# pygame.init()

# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Simple Game")

# clock = pygame.time.Clock()
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def reset_game():
    board = np.zeros((13, 13))
    snake_direction = [1, 0]

    snake = spawn_snake(board) # set head as 1, body as 2
    spawn_food(board) # set food as 3   
    
    return np.concatenate((board.flatten(), snake_direction)), snake
    

def step_game(action, state, snake):
    board = state[0][:169].reshape(13, 13)
    direction = state[0][169:]
    
    direction = [1, 0]
    
    if action == 0: 
        new_direction = direction
    elif action == 1:  # Turn left
        new_direction = [direction[1], -direction[0]]  # Rotate left
    elif action == 2:  # Turn right
        new_direction = [-direction[1], direction[0]]  # Rotate left
        
    head = np.argwhere(board == 1)[0]
    new_head = head + new_direction
    
    if new_head[0] < 0 or new_head[0] >= 13 or new_head[1] < 0 or new_head[1] >= 13 or board[new_head[0], new_head[1]] == 2:
        return state, -1, True, snake
    
    if board[new_head[0], new_head[1]] == 3:
        reward = 15
        snake.append(snake[-1])
    else:
        reward = -0.01
    
    board[new_head[0], new_head[1]] = 1
    old_tail = snake[-1]
    board[old_tail[0], old_tail[1]] = 0
    for i in range(len(snake) - 1, 0, -1):
        snake[i] = snake[i - 1]
    snake[0] = new_head
    for i in range(1, len(snake)):
        board[snake[i][0], snake[i][1]] = 2
    
    return np.concatenate((board.flatten(), new_direction)), reward, False, snake
    


def spawn_food(board):
    food_count = np.argwhere(board == 3).shape[0]
    if food_count >= 3:
        return False

    empty = np.argwhere(board == 0)
    if len(empty) == 0:
        return False

    for _ in range(3 - food_count):
        idx = np.random.randint(len(empty))
        board[empty[idx][0], empty[idx][1]] = 3
        empty = np.delete(empty, idx, axis=0)
        
    return True

def spawn_snake(board):
    
    # any where from 4 to 8
    x = np.random.randint(4, 9)
    y = np.random.randint(4, 9)
    
    board[x, y] = 1
    board[x - 1, y] = 2
    board[x - 2, y] = 2
    return [[x, y], [x - 1, y], [x - 2, y]]


# def draw_board(board, screen):
#     screen.fill((0, 0, 0))
#     for i in range(13):
#         for j in range(13):
#             if board[i, j] == 1:  # head
#                 pygame.draw.rect(screen, (255, 0, 0), (i * 50, j * 50, 50, 50))
#             elif board[i, j] == 2:  # body
#                 pygame.draw.rect(screen, (0, 255, 0), (i * 50, j * 50, 50, 50))
#             elif board[i, j] == 3:  # food
#                 pygame.draw.rect(screen, (0, 0, 255), (i * 50, j * 50, 50, 50))
                
#     pygame.display.flip()


def spawn_food(board):
    if np.argwhere(board == 3).shape[0] > 0:
        return False

    empty = np.argwhere(board == 0)
    if len(empty) == 0:
        return False
    idx = np.random.randint(len(empty))
    board[empty[idx][0], empty[idx][1]] = 3
    return True





def build_model():
    input_n = 13*13 + 2
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_dim=input_n, activation='relu', name="input"),
        tf.keras.layers.Dense(64, activation='relu', name="hidden1"),
        tf.keras.layers.Dense(3, activation='linear', name="output")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')

    return model


replay_buffer = deque(maxlen=2000)
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.1
gamma = 0.95  # Discount factor
batch_size = 64


model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())  # Target network


def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(3)  # Explore: random action
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])  # Exploit: best action


def train():
    if len(replay_buffer) < batch_size:
        return
    minibatch = random.sample(replay_buffer, batch_size)

    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.max(target_model.predict(next_state, verbose=0)[0])
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)


def update_target_model():
    target_model.set_weights(model.get_weights())



for episode in range(1000):  # Number of episodes
    state, snake = reset_game()  # Initialize game state
    # print(f"Episode {episode+1}")
    # print(snake)
    state = np.reshape(state, [1, 171])
    score = 0

    for time in range(250):  # Max steps per episode
        action = choose_action(state, epsilon)
        next_state, reward, done, new_snake = step_game(action, state, snake)
        next_state = np.reshape(next_state, [1, 171])
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        snake = new_snake
        score += reward

        # draw_board(state[:169].reshape(13, 13), screen)
        
        if done:
            print(f"Episode {episode+1}: Score {score}")
            break

        train()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 10 == 0:
        update_target_model()
        
    if episode % 50 == 0:
        model.save(f"snake_model_3_food_{episode}.h5")
    
model.save("snake_model.h5")
