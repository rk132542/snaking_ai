import pygame
import numpy as np

WIDTH, HEIGHT = 650, 650
FPS = 6

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Game")

clock = pygame.time.Clock()

def spawn_snake(board, x, y):
    board[x, y] = 1
    board[x - 1, y] = 2
    board[x - 2, y] = 2
    return [[x, y], [x - 1, y], [x - 2, y]]

def draw_board(board, screen):
    for i in range(13):
        for j in range(13):
            if board[i, j] == 1: # head
                pygame.draw.rect(screen, (255, 0, 0), (i * 50, j * 50, 50, 50))
            elif board[i, j] == 2: # body
                pygame.draw.rect(screen, (0, 255, 0), (i * 50, j * 50, 50, 50))
            elif board[i, j] == 3: # food
                pygame.draw.rect(screen, (0, 0, 255), (i * 50, j * 50, 50, 50))

def spawn_food(board):
    if np.argwhere(board == 3).shape[0] > 0:
        return False
    
    empty = np.argwhere(board == 0)
    if len(empty) == 0:
        return False
    idx = np.random.randint(len(empty))
    board[empty[idx][0], empty[idx][1]] = 3
    return True

def main():
    board = np.zeros((13, 13)) # 13x13 board, 0 is empty, 1 is head, 2 is body, 3 is food
    snake_direction = [1,0] # start with right
    snake_length = 3
    
    snake = spawn_snake(board, 5, 5)
    running = True
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake_direction[1] == 0:
                    snake_direction = [0, -1]
                elif event.key == pygame.K_DOWN and snake_direction[1] == 0:
                    snake_direction = [0, 1]
                elif event.key == pygame.K_LEFT and snake_direction[0] == 0:
                    snake_direction = [-1, 0]
                elif event.key == pygame.K_RIGHT and snake_direction[0] == 0:
                    snake_direction = [1, 0]
            if event.type == pygame.QUIT:
                running = False

        # update snake
        
        spawn_food(board)
        
        head = np.argwhere(board == 1)[0]
        new_head = head + snake_direction
        if new_head[0] < 0 or new_head[0] >= 13 or new_head[1] < 0 or new_head[1] >= 13:
            running = False
            break
        elif board[new_head[0], new_head[1]] == 2:
            running = False
            break
        else:
            if board[new_head[0], new_head[1]] == 3:
                snake.append(snake[-1])
                snake_length += 1
            
            board[new_head[0], new_head[1]] = 1
            old_tail = snake[-1]
            board[old_tail[0], old_tail[1]] = 0
            for i in range(len(snake) - 1, 0, -1):
                snake[i] = snake[i - 1]
            snake[0] = new_head
            for i in range(1, len(snake)):
                board[snake[i][0], snake[i][1]] = 2
        
        print(snake[0])
        draw_board(board, screen)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()