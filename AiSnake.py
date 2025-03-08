import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

BLOCK_SIZE = 4  # Уменьшили размер блока для карты 250x250
GRID_SIZE = 250  # Увеличили поле до 250x250
WIDTH = BLOCK_SIZE * GRID_SIZE
HEIGHT = BLOCK_SIZE * GRID_SIZE
FPS = 15
COLOR_BG = (10, 10, 10)
COLOR_GRID = (20, 20, 20)
COLOR_SNAKE = (0, 255, 0)
COLOR_APPLE = (255, 0, 0)  # Красные яблоки
COLOR_GOLD_APPLE = (255, 215, 0)  # Золотые яблоки
NUM_RED_APPLES = 50  # Количество красных яблок
NUM_GOLD_APPLES = 5  # Количество золотых яблок
MIN_APPLE_TIMER = 50  # Минимальное время жизни яблока
MAX_APPLE_TIMER = 200  # Максимальное время жизни яблока

class SnakeGameAI:
    def __init__(self, w=WIDTH, h=HEIGHT, display=True):
        self.w = w
        self.h = h
        self.display = display
        if self.display:
            pygame.init()
            self.display_surface = pygame.display.set_mode((w, h))
            pygame.display.set_caption('Змейка + ИИ')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)  # Начальное направление змейки
        self.head = [GRID_SIZE // 2, GRID_SIZE // 2]  # Начальная позиция головы
        self.snake = [self.head.copy()]  # Тело змейки
        self.score = 0  # Счет
        self.red_apples = []  # Список красных яблок
        self.gold_apples = []  # Список золотых яблок
        self.red_apple_timers = []  # Таймеры для красных яблок
        self.gold_apple_timers = []  # Таймеры для золотых яблок
        self._place_apples()  # Размещение яблок
        self.frame_iteration = 0  # Счетчик итераций

    def _place_apples(self):
        # Размещаем красные яблоки
        self.red_apples = []
        self.red_apple_timers = []
        for _ in range(NUM_RED_APPLES):
            while True:
                x = random.randint(0, GRID_SIZE - 1)
                y = random.randint(0, GRID_SIZE - 1)
                if [x, y] not in self.snake and [x, y] not in self.red_apples and [x, y] not in self.gold_apples:
                    self.red_apples.append([x, y])
                    self.red_apple_timers.append(random.randint(MIN_APPLE_TIMER, MAX_APPLE_TIMER))
                    break

        # Размещаем золотые яблоки
        self.gold_apples = []
        self.gold_apple_timers = []
        for _ in range(NUM_GOLD_APPLES):
            while True:
                x = random.randint(0, GRID_SIZE - 1)
                y = random.randint(0, GRID_SIZE - 1)
                if [x, y] not in self.snake and [x, y] not in self.red_apples and [x, y] not in self.gold_apples:
                    self.gold_apples.append([x, y])
                    self.gold_apple_timers.append(random.randint(MIN_APPLE_TIMER, MAX_APPLE_TIMER))
                    break

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)  # Движение змейки
        self.snake.insert(0, self.head.copy())  # Добавление новой головы
        reward = 0
        game_over = False

        # Проверка на столкновение или превышение лимита итераций
        if self._is_collision() or self.frame_iteration > 500 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Проверка, съела ли змейка красное яблоко
        for i, apple in enumerate(self.red_apples):
            if self.head == apple:
                self.score += 1
                reward = 10
                # Перемещаем яблоко в новую позицию
                while True:
                    new_x = random.randint(0, GRID_SIZE - 1)
                    new_y = random.randint(0, GRID_SIZE - 1)
                    if [new_x, new_y] not in self.snake and [new_x, new_y] not in self.red_apples and [new_x, new_y] not in self.gold_apples:
                        self.red_apples[i] = [new_x, new_y]
                        break
                break

        # Проверка, съела ли змейка золотое яблоко
        for i, apple in enumerate(self.gold_apples):
            if self.head == apple:
                self.score += 5  # Золотое яблоко дает больше очков
                reward = 50
                # Перемещаем яблоко в новую позицию
                while True:
                    new_x = random.randint(0, GRID_SIZE - 1)
                    new_y = random.randint(0, GRID_SIZE - 1)
                    if [new_x, new_y] not in self.snake and [new_x, new_y] not in self.red_apples and [new_x, new_y] not in self.gold_apples:
                        self.gold_apples[i] = [new_x, new_y]
                        break
                break
        else:
            self.snake.pop()  # Удаление хвоста, если яблоко не съедено

        # Обновление таймеров красных яблок
        for i in range(len(self.red_apple_timers)):
            self.red_apple_timers[i] -= 1
            if self.red_apple_timers[i] <= 0:
                # Перемещаем яблоко в новую позицию
                while True:
                    new_x = random.randint(0, GRID_SIZE - 1)
                    new_y = random.randint(0, GRID_SIZE - 1)
                    if [new_x, new_y] not in self.snake and [new_x, new_y] not in self.red_apples and [new_x, new_y] not in self.gold_apples:
                        self.red_apples[i] = [new_x, new_y]
                        self.red_apple_timers[i] = random.randint(MIN_APPLE_TIMER, MAX_APPLE_TIMER)
                        break

        # Обновление таймеров золотых яблок
        for i in range(len(self.gold_apple_timers)):
            self.gold_apple_timers[i] -= 1
            if self.gold_apple_timers[i] <= 0:
                # Перемещаем яблоко в новую позицию
                while True:
                    new_x = random.randint(0, GRID_SIZE - 1)
                    new_y = random.randint(0, GRID_SIZE - 1)
                    if [new_x, new_y] not in self.snake and [new_x, new_y] not in self.red_apples and [new_x, new_y] not in self.gold_apples:
                        self.gold_apples[i] = [new_x, new_y]
                        self.gold_apple_timers[i] = random.randint(MIN_APPLE_TIMER, MAX_APPLE_TIMER)
                        break

        if self.display:
            self._update_ui()
            self.clock.tick(FPS)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Столкновение с границами
        if pt[0] < 0 or pt[0] >= GRID_SIZE or pt[1] < 0 or pt[1] >= GRID_SIZE:
            return True
        # Столкновение с телом
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display_surface.fill(COLOR_BG)
        # Отрисовка сетки
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display_surface, COLOR_GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display_surface, COLOR_GRID, (0, y), (self.w, y))
        # Отрисовка красных яблок
        for apple in self.red_apples:
            self._draw_block(COLOR_APPLE, apple)
        # Отрисовка золотых яблок
        for apple in self.gold_apples:
            self._draw_block(COLOR_GOLD_APPLE, apple)
        # Отрисовка змейки
        for pt in self.snake:
            self._draw_block(COLOR_SNAKE, pt)
        # Отрисовка счета
        self._draw_score()
        pygame.display.flip()

    def _draw_block(self, color, pos):
        rect = pygame.Rect(pos[0] * BLOCK_SIZE, pos[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display_surface, color, rect)

    def _draw_score(self):
        font = pygame.font.SysFont('arial', 25)
        score_surface = font.render(f'Score: {self.score}', True, (200, 200, 200))
        self.display_surface.blit(score_surface, (10, 10))

    def _move(self, action):
        clock_wise = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Направления: вверх, вправо, вниз, влево
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):  # Движение прямо
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Поворот направо
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):  # Поворот налево
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = clock_wise[idx]
        self.direction = new_dir
        self.head[0] += self.direction[0]
        self.head[1] += self.direction[1]

# ии для обучения
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# Основной класс для ии
class SnakeAI:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.0005, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]
        dir_l = (game.direction == (-1, 0))
        dir_r = (game.direction == (1, 0))
        dir_u = (game.direction == (0, -1))
        dir_d = (game.direction == (0, 1))

        danger_straight = (dir_r and game._is_collision(point_r)) or \
                          (dir_l and game._is_collision(point_l)) or \
                          (dir_u and game._is_collision(point_u)) or \
                          (dir_d and game._is_collision(point_d))

        danger_right = (dir_u and game._is_collision(point_r)) or \
                       (dir_d and game._is_collision(point_l)) or \
                       (dir_l and game._is_collision(point_u)) or \
                       (dir_r and game._is_collision(point_d))

        danger_left = (dir_d and game._is_collision(point_r)) or \
                      (dir_u and game._is_collision(point_l)) or \
                      (dir_r and game._is_collision(point_u)) or \
                      (dir_l and game._is_collision(point_d))

        # Направление движения
        direction_left = dir_l
        direction_right = dir_r
        direction_up = dir_u
        direction_down = dir_d

        # Положение ближайшего яблока
        apple_positions = game.red_apples + game.gold_apples
        if apple_positions:
            closest_apple = min(apple_positions, key=lambda x: abs(x[0] - head[0]) + abs(x[1] - head[1]))
            apple_left = closest_apple[0] < head[0]
            apple_right = closest_apple[0] > head[0]
            apple_up = closest_apple[1] < head[1]
            apple_down = closest_apple[1] > head[1]
        else:
            apple_left = apple_right = apple_up = apple_down = 0

        state = [
            danger_straight, danger_right, danger_left,
            direction_left, direction_right, direction_up, direction_down,
            apple_left, apple_right, apple_up, apple_down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, 1000) if len(self.memory) > 1000 else self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

# Функция для обучения
def train():
    scores = []
    total_score = 0
    record = 0
    snake_ai = SnakeAI()
    game = SnakeGameAI(display=True)

    while True:
        state_old = snake_ai.get_state(game)
        final_move = snake_ai.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = snake_ai.get_state(game)
        snake_ai.train_short_memory(state_old, final_move, reward, state_new, done)
        snake_ai.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            snake_ai.n_games += 1
            snake_ai.train_long_memory()

            # Обновление статистики
            scores.append(score)
            total_score += score
            mean_score = total_score / snake_ai.n_games

            # Обновление рекорда
            if score > record:
                record = score

            # Вывод статистики в консоль
            print(f'Игра {snake_ai.n_games}: Счет = {score}, Рекорд = {record}, Средний счет = {mean_score:.2f}')

if __name__ == '__main__':
    train()