import pygame
import sys
import numpy as np
from thud_game import ThudGame, DWARF, TROLL, THUDSTONE  # убедитесь, что файл thud_game.py находится в той же директории

# Конфигурация графики
CELL_SIZE = 40
BOARD_ROWS = 15
BOARD_COLS = 15
WIDTH = CELL_SIZE * BOARD_COLS
HEIGHT = CELL_SIZE * BOARD_ROWS

# Определение цветов
COLOR_BG = (240, 240, 240)
COLOR_GRID = (200, 200, 200)
COLOR_INVALID = (50, 50, 50)
COLOR_EMPTY = (220, 220, 220)
COLOR_DWARF = (139, 69, 19)       # коричневый для дворфов
COLOR_TROLL = (178, 34, 34)        # красноватый для троллей
COLOR_THUDSTONE = (0, 0, 0)        # чёрный для Thudstone
COLOR_HIGHLIGHT = (0, 255, 0)      # зелёный для подсветки допустимых ходов
COLOR_CAPTURE = (229, 43, 80)
COLOR_LINE = (218, 165, 32)
COLOR_SELECTED = (255, 215, 0)     # золотой для выделенной фигуры

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + 40))  # дополнительное пространство для вывода статуса
pygame.display.set_caption("THUD! Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

def draw_board(game, selected=None, legal_moves=None, line_mask=None, capture_mask=None):
    screen.fill(COLOR_BG)
    
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # Если клетка недопустимая, закрашиваем её в тёмный цвет
            if not game.valid_mask[row, col]:
                pygame.draw.rect(screen, COLOR_INVALID, rect)
                continue
            else:
                pygame.draw.rect(screen, COLOR_EMPTY, rect)
            
            # Рисуем сетку
            pygame.draw.rect(screen, COLOR_GRID, rect, 1)
            
            # Подсветка допустимых ходов
            if legal_moves is not None and legal_moves[row, col] == 1:
                if capture_mask is not None and capture_mask[row, col] > 0:
                    pygame.draw.rect(screen, COLOR_CAPTURE, rect, 3)
                elif line_mask is not None and line_mask[row, col] == 1:
                    pygame.draw.rect(screen, COLOR_LINE, rect, 3)
                else:
                    pygame.draw.rect(screen, COLOR_HIGHLIGHT, rect, 3)
            
            # Рисуем фигуры
            piece = game.board[row, col]
            center = (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2)
            radius = CELL_SIZE // 2 - 4
            if piece == DWARF:
                pygame.draw.circle(screen, COLOR_DWARF, center, radius)
            elif piece == TROLL:
                pygame.draw.circle(screen, COLOR_TROLL, center, radius)
            elif piece == THUDSTONE:
                pygame.draw.circle(screen, COLOR_THUDSTONE, center, radius)
                
            # Если клетка выбрана, выделяем её рамкой
            if selected is not None and selected == (row, col):
                pygame.draw.rect(screen, COLOR_SELECTED, rect, 3)
    
    # Выводим статус (текущий игрок, номер хода)
    status_text = f"Ход: {game.current_player.upper()}   Ходов: {game.move_count}/{game.move_limit}"
    text_surface = font.render(status_text, True, (0, 0, 0))
    screen.blit(text_surface, (5, HEIGHT + 5))

def main():
    game = ThudGame(move_limit=25)
    selected_unit = None
    legal_moves = None
    
    running = True
    while running:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                # Обработка клика по игровому полю
                if mouse_y <= HEIGHT:
                    row = mouse_y // CELL_SIZE
                    col = mouse_x // CELL_SIZE
                    # Если фигура ещё не выбрана, пытаемся её выбрать
                    if selected_unit is None:
                        if game.legal_units()[row, col] == 1:
                            selected_unit = (row, col)
                            legal_moves = game.get_legal_moves(selected_unit)
                    else:
                        # Если фигура выбрана, проверяем, является ли клик по допустимому ходу
                        if legal_moves is not None and legal_moves[row, col] == 1:
                            try:
                                game.make_move(selected_unit, (row, col))
                            except ValueError as e:
                                print("Ошибка хода:", e)
                            selected_unit = None
                            legal_moves = None
                        else:
                            # Если клик по другой фигуре, пытаемся выбрать её
                            if game.legal_units()[row, col] == 1:
                                selected_unit = (row, col)
                                legal_moves = game.get_legal_moves(selected_unit)
                            else:
                                selected_unit = None
                                legal_moves = None
        
        if game.current_player == 'dwarf':
            line_mask = game.dwarf_line_mask(legal_moves, selected_unit)
            capture_mask = game.dwarf_capture_tile(legal_moves)
        else:
            line_mask = game.troll_line_mask(legal_moves, selected_unit)
            capture_mask = game.troll_capture_tile(legal_moves)

        draw_board(game, selected=selected_unit, legal_moves=legal_moves, line_mask=line_mask, capture_mask=capture_mask)
        pygame.display.flip()
        
        # Если игра окончена, выводим сообщение и ждем нажатия клавиши для выхода
        if game.is_game_over():
            winner = game.get_winner()
            print("Игра окончена. Победитель:", winner)
            msg = f"Игра окончена. Победитель: {winner.upper()}. Нажмите любую клавишу для выхода."
            text_surface = font.render(msg, True, (0, 0, 0))
            screen.blit(text_surface, (5, HEIGHT + 25))
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        waiting = False
                        running = False

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
