from typing import Optional
import numpy as np

# Определяем константы для обозначения клеток
EMPTY = 0
DWARF = 1
TROLL = 2
THUDSTONE = 3
INVALID = -1  # для неигровых клеток

# Направления (все 8 направлений)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

class ThudGame:
    def __init__(self, move_limit=400, no_catrure_limit=200):
        self.move_limit = move_limit
        self.no_catrure_limit = no_catrure_limit
        self.move_count = 0
        self.no_catrure_count = 0
        self.current_player = 'dwarf'  # дворфы ходят первыми
        
        # Создаём игровое поле 15x15
        self.board = np.full((15, 15), INVALID, dtype=int)
        self.valid_mask = self.generate_valid_mask()
        self.board[self.valid_mask] = EMPTY
        
        # Устанавливаем Thudstone в центр (позиция 7,7)
        self.board[7, 7] = THUDSTONE
        
        # Расставляем троллей – восемь клеток вокруг центра
        troll_positions = [(6, 6), (6, 7), (6, 8),
                           (7, 6),         (7, 8),
                           (8, 6), (8, 7), (8, 8)]
        for pos in troll_positions:
            self.board[pos] = TROLL
        
        # Расставляем дворфов по периметру допустимых клеток,
        # исключая 4 клетки, выровненные по горизонтали/вертикали с центром
        perimeter = self.get_perimeter_positions()
        exclude = {(0, 7), (7, 0), (7, 14), (14, 7)}
        for pos in perimeter:
            if pos not in exclude:
                self.board[pos] = DWARF

    def generate_valid_mask(self):
        """
        Создаёт булеву маску 15x15, определяющую допустимые (игровые) клетки.
        На доске удалены в каждом углу треугольники из 15 клеток (1+2+3+4+5).
        """
        mask = np.ones((15, 15), dtype=bool)
        for r in range(15):
            for c in range(15):
                # Верхний левый угол
                if r < 5 and c < 5 - r:
                    mask[r, c] = False
                # Верхний правый угол
                if r < 5 and c >= 15 - (5 - r):
                    mask[r, c] = False
                # Нижний левый угол
                if r >= 10 and c < r - 9:
                    mask[r, c] = False
                # Нижний правый угол
                if r >= 10 and c >= 15 - (r - 9):
                    mask[r, c] = False
        return mask

    def in_bounds(self, r, c):
        return 0 <= r < 15 and 0 <= c < 15

    def get_perimeter_positions(self):
        """
        Возвращает список координат (r, c) допустимых клеток,
        находящихся на периметре (то есть имеющих хотя бы одного соседнего неигрового поля).
        """
        perimeter = []
        for r in range(15):
            for c in range(15):
                if self.valid_mask[r, c]:
                    is_edge = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == dc or dr == -dc:
                                continue
                            nr, nc = r + dr, c + dc
                            if not self.in_bounds(nr, nc) or not self.valid_mask[nr, nc]:
                                is_edge = True
                                break
                        if is_edge:
                            perimeter.append((r, c))
                            break
        return perimeter

    def legal_units(self):
        """
        Возвращает one-hot матрицу 15x15 с теми клетками, в которых находятся
        фигуры текущего игрока, имеющие хотя бы один допустимый ход.
        """
        legal = np.zeros((15, 15), dtype=int)
        piece_type = DWARF if self.current_player == 'dwarf' else TROLL
        positions = np.argwhere(self.board == piece_type)
        for pos in positions:
            pos = tuple(pos)
            moves = self.get_legal_moves(pos)
            if np.sum(moves) > 0:
                legal[pos] = 1
        return legal
    
    def dwarf_line_mask(self, moves, pos):
        mask = np.zeros((15, 15), dtype=int)
        if moves is None or pos is None or self.current_player == 'troll':
            return mask
        for r, row in enumerate(moves):
            for c in range(len(row)):
                if moves[r][c]:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if self.in_bounds(nr, nc) and self.valid_mask[nr, nc] and self.board[nr, nc] == DWARF and pos != (nr, nc):
                                mask[r][c] = 1
        return mask
    
    def troll_line_mask(self, moves, pos):
        mask = np.zeros((15, 15), dtype=int)
        if moves is None or pos is None or self.current_player == 'dwarf':
            return mask
        for r, row in enumerate(moves):
            for c in range(len(row)):
                if moves[r][c]:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if self.in_bounds(nr, nc) and self.valid_mask[nr, nc] and self.board[nr, nc] == TROLL and pos != (nr, nc):
                                mask[r][c] = 1
        return mask

    def dwarf_capture_tile(self, moves):
        mask = np.zeros((15, 15), dtype=int)
        if moves is None or self.current_player == 'troll':
            return mask
        for r, row in enumerate(moves):
            for c in range(len(row)):
                if moves[r][c]:
                    if self.in_bounds(r, c) and self.valid_mask[r, c] and self.board[r, c] == TROLL:
                        mask[r][c] = 1
        return mask

    def troll_capture_tile(self, moves):
        mask = np.zeros((15, 15), dtype=float)
        if moves is None or self.current_player == 'dwarf':
            return mask
        for r, row in enumerate(moves):
            for c in range(len(row)):
                if moves[r][c]:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if self.in_bounds(nr, nc) and self.valid_mask[nr, nc] and self.board[nr, nc] == DWARF:
                                mask[r][c] += 1
        return mask / 8
    
    def player_scalar(self):
        return 0 if self.current_player == 'dwarf' else 1
    
    def phase_scalar(self, pos: Optional[tuple[int, int]] = None):
        return 0 if pos is None else 1

    def starvation_scalar(self):
        return (self.move_limit - self.move_count) / self.move_limit
    
    def starvation_no_catrures(self):
        pass
    
    def field_tile(self, h: Optional[int] = None, w: Optional[int] = None):
        if h is None:
            h = self.board.shape[0]
        if w is None:
            w = self.board.shape[1]
        categories = [DWARF, TROLL, THUDSTONE]
        units_one_hot = np.array([(self.board == category) for category in categories], dtype=int)
        units_one_hot = np.pad(units_one_hot, ((0, 0), (0, h - units_one_hot.shape[1]), (0, w - units_one_hot.shape[2])))
        valid_mask = np.pad(self.valid_mask, ((0, h - self.valid_mask.shape[0]), (0, w - self.valid_mask.shape[1])))
        return np.concatenate((valid_mask[None,], units_one_hot), axis=0)

    def selected_pice_tile(self, pos: Optional[tuple[int, int]] = None, h: Optional[int] = None, w: Optional[int] = None):
        if h is None:
            h = self.board.shape[0]
        if w is None:
            w = self.board.shape[1]
        tile = np.zeros((h, w), dtype=float)
        if pos is not None:
            tile[pos] = 1
        return tile[None,]

    def dwarf_value_scalar(self):
        print('dwarfs count', (self.board == DWARF).sum())
        return (self.board == DWARF).sum() / 32

    def troll_value_scalar(self):
        print('troll count', (self.board == TROLL).sum())
        return (self.board == TROLL).sum() * 4 / 32

    def scalar_to_tile(self, scalar, h: Optional[int] = None, w: Optional[int] = None):
        if h is None:
            h = self.board.shape[0]
        if w is None:
            w = self.board.shape[1]
        tile = np.zeros((h, w), dtype=float)
        tile[:] = scalar
        return tile[None,]
    
    def pad_tile(self, tile, h: Optional[int] = None, w: Optional[int] = None):
        if h is None:
            h = self.board.shape[0]
        if w is None:
            w = self.board.shape[1]
        return np.pad(tile, ((0, h - tile.shape[0]), (0, w - tile.shape[1])))[None,]

    
    def observations(self, pos: Optional[tuple[int, int]] = None, h: Optional[int] = None, w: Optional[int] = None):
        if h is None:
            h = self.board.shape[0]
        if w is None:
            w = self.board.shape[1]
        
        legal_units = self.legal_units()
        if pos is not None:
            legal_moves = self.get_legal_moves(pos)
        else:
            legal_moves = np.zeros((h, w), dtype=float)

        if self.current_player == 'dwarf':
            line_mask = self.dwarf_line_mask(legal_moves, pos)
            capture_tile = self.dwarf_capture_tile(legal_moves)
        else:
            line_mask = self.troll_line_mask(legal_moves, pos)
            capture_tile = self.troll_capture_tile(legal_moves)

        data = np.concatenate([
            self.field_tile(h, w),
            self.selected_pice_tile(pos, h, w),
            self.pad_tile(legal_units, h, w),
            self.pad_tile(legal_moves, h, w),
            self.pad_tile(line_mask, h, w),
            self.pad_tile(capture_tile, h, w),
            self.scalar_to_tile(self.starvation_scalar(), h, w),
            self.scalar_to_tile(self.dwarf_value_scalar(), h, w),
            self.scalar_to_tile(self.troll_value_scalar(), h, w),
        ]).astype(np.float32)
        return data

    def get_legal_moves(self, pos: Optional[tuple[int, int]] = None):
        """
        Для выбранной фигуры (pos) возвращает one-hot матрицу 15x15,
        где 1 отмечены клетки, куда можно пойти.
        Реализованы как:
          - Для дворфов: перемещение как шахматная ферзь + "метание" для захвата тролля.
          - Для троллей: перемещение как шахматный король + возможность штовхания (если захватываются дворфы).
        """
        moves = np.zeros((15, 15), dtype=int)
        if pos is None:
            return moves
        r, c = pos
        if self.current_player == 'dwarf':
            if self.board[r, c] != DWARF:
                return moves
            # Обычное перемещение (как ферзь)
            for dr, dc in DIRECTIONS:
                step = 1
                while True:
                    nr, nc = r + step * dr, c + step * dc
                    if not self.in_bounds(nr, nc) or not self.valid_mask[nr, nc]:
                        break
                    if self.board[nr, nc] != EMPTY:
                        break
                    moves[nr, nc] = 1
                    step += 1

            # Специальное перемещение – метание (hurl)
            # Фигура может быть метнута, если она является "передней" в линии соседних дворфов
            for dr, dc in DIRECTIONS:
                # Проверяем, что с обратной стороны нет дворфа – значит, мы на передней границе линии
                br, bc = r - dr, c - dc
                if self.in_bounds(br, bc) and self.board[br, bc] == DWARF:
                    continue
                # Считаем длину непрерывной линии дворфов в направлении (dr, dc)
                L = 0
                tr, tc = r, c
                while self.in_bounds(tr, tc) and self.valid_mask[tr, tc] and self.board[tr, tc] == DWARF:
                    L += 1
                    tr += dr
                    tc += dc
                # Если следующая клетка содержит тролля, то ход метанием возможен
                if L == 1:
                    continue
                for d in range(1, L + 1):
                    tr = r - d * dr
                    tc = c - d * dc
                    if self.in_bounds(tr, tc) and self.valid_mask[tr, tc] and self.board[tr, tc] == TROLL:
                        moves[tr, tc] = 1
                        break

            return moves

        elif self.current_player == 'troll':
            if self.board[r, c] != TROLL:
                return moves
            # Обычное перемещение (как король)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc) and self.valid_mask[nr, nc] and self.board[nr, nc] == EMPTY:
                        moves[nr, nc] = 1

            # Штовхание (shove)
            # Если тролль является "передним" в линии троллей, то он может быть штовхнут на 1..L клеток
            for dr, dc in DIRECTIONS:
                br, bc = r - dr, c - dc
                if self.in_bounds(br, bc) and self.board[br, bc] == TROLL:
                    continue
                L = 0
                tr, tc = r, c
                while self.in_bounds(tr, tc) and self.valid_mask[tr, tc] and self.board[tr, tc] == TROLL:
                    L += 1
                    tr += dr
                    tc += dc
                for d in range(1, L + 1):
                    nr = r - d * dr
                    nc = c - d * dc
                    if not self.in_bounds(nr, nc) or not self.valid_mask[nr, nc]:
                        break
                    if self.board[nr, nc] != EMPTY:
                        break
                    # Ход штовхивания разрешён, если при перемещении хотя бы один сосед (из 8) содержит дворфа
                    captured = False
                    for adj_dr in [-1, 0, 1]:
                        for adj_dc in [-1, 0, 1]:
                            if adj_dr == 0 and adj_dc == 0:
                                continue
                            ar, ac = nr + adj_dr, nc + adj_dc
                            if self.in_bounds(ar, ac) and self.valid_mask[ar, ac] and self.board[ar, ac] == DWARF:
                                captured = True
                    if captured:
                        moves[nr, nc] = 1
            
            return moves

        else:
            return moves

    def make_move(self, from_pos, to_pos, move_type=None):
        """
        Выполняет ход:
          - Для дворфов: обычное перемещение или метание (hurl), при котором тролль захватывается.
          - Для троллей: обычное перемещение (с опциональным захватом) или штовхание (shove).
        После хода происходит переключение игрока и увеличение счётчика ходов.
        Если достигнут лимит ходов (move_limit), игра считается ничьей.
        """
        legal = self.get_legal_moves(from_pos)
        if legal[to_pos] != 1:
            raise ValueError("Неверный ход: выбранное перемещение недопустимо.")

        if self.current_player == 'dwarf':
            # Если ход не указан явно, определяем его тип:
            dr = to_pos[0] - from_pos[0]
            dc = to_pos[1] - from_pos[1]
            steps = max(abs(dr), abs(dc))
            if steps == 1 and self.board[to_pos] == TROLL:
                move_type = 'hurl'
            else:
                move_type = 'move' if self.board[to_pos] == EMPTY else 'hurl'

            if move_type == 'move':
                self.board[to_pos] = DWARF
                self.board[from_pos] = EMPTY
                self.no_catrure_count += 1
            elif move_type == 'hurl':
                # При метании тролль снимается с доски, а дворф занимает его клетку
                self.board[to_pos] = DWARF
                self.board[from_pos] = EMPTY
                self.no_catrure_count = 0
            else:
                raise ValueError("Неизвестный тип хода для дворфов.")
        else:  # ход тролля
            # Если ход не указан, определяем его по расстоянию
            dr = to_pos[0] - from_pos[0]
            dc = to_pos[1] - from_pos[1]
            if abs(dr) <= 1 and abs(dc) <= 1:
                move_type = 'move'
            else:
                move_type = 'shove'

            if move_type == 'move':
                self.board[to_pos] = TROLL
                self.board[from_pos] = EMPTY
                # Дополнительный опциональный захват: снимается один соседний дворф (если есть)
                for adj_dr in [-1, 0, 1]:
                    for adj_dc in [-1, 0, 1]:
                        if adj_dr == 0 and adj_dc == 0:
                            continue
                        nr, nc = to_pos[0] + adj_dr, to_pos[1] + adj_dc
                        # TODO: какого дворфа захватить, должен решать игрок
                        if self.in_bounds(nr, nc) and self.board[nr, nc] == DWARF:
                            self.board[nr, nc] = EMPTY
                            self.no_catrure_count = 0
                            break
                    else:
                        continue
                    break
                if self.no_catrure_count != 0:
                    self.no_catrure_count += 1
            elif move_type == 'shove':
                # Штовхание: перемещаем тролля по направлению, затем захватываем всех соседних дворфов
                dr = np.sign(to_pos[0] - from_pos[0])
                dc = np.sign(to_pos[1] - from_pos[1])
                self.board[to_pos] = TROLL
                self.board[from_pos] = EMPTY
                for adj_dr in [-1, 0, 1]:
                    for adj_dc in [-1, 0, 1]:
                        if adj_dr == 0 and adj_dc == 0:
                            continue
                        nr, nc = to_pos[0] + adj_dr, to_pos[1] + adj_dc
                        if self.in_bounds(nr, nc) and self.board[nr, nc] == DWARF:
                            self.board[nr, nc] = EMPTY
                self.no_catrure_count = 0
            else:
                raise ValueError("Неизвестный тип хода для тролля.")
                
        self.move_count += 1
        # Переключаем игрока после каждого хода
        self.current_player = 'troll' if self.current_player == 'dwarf' else 'dwarf'
        print(np.sum(self.board == TROLL), np.sum(self.board == DWARF))

    def is_game_over(self):
        """
        Проверяет, окончена ли игра.
        Игра завершается, если:
          - У текущего игрока нет допустимых ходов, или
          - Достигнут лимит ходов (например, 400 ходов).
        """
        if np.sum(self.legal_units()) == 0:
            return True
        if self.move_count >= self.move_limit:
            return True
        return False

    def get_winner(self):
        """
        Простой вариант определения победителя.
        Можно, например, подсчитать оставшихся фигур:
          - Дворфы получают 1 очко за каждого выжившего,
          - Тролли – 4 очка за каждого.
        Если подсчет невозможен (например, ничья по лимиту ходов), возвращается 'draw'.
        """
        dwarf_count = np.sum(self.board == DWARF)
        troll_count = np.sum(self.board == TROLL)
        score = dwarf_count - 4 * troll_count
        if score > 0:
            return 'dwarf'
        elif score < 0:
            return 'troll'
        else:
            return 'draw'

# Пример использования:
if __name__ == '__main__':
    game = ThudGame(move_limit=400)
    # print("Начальное поле:")
    # print(game.board)
    # print("Legal units (для текущего игрока):")
    # print(game.legal_units())
    # # Для примера: выбираем первую найденную фигуру текущего игрока и выводим её допустимые ходы
    # units = np.argwhere(game.legal_units() == 1)
    # if units.size > 0:
    #     selected = tuple(units[0])
    #     print("Выбранная фигура в позиции:", selected)
    #     print("Legal moves для выбранной фигуры:")
    #     print(game.get_legal_moves(selected))
    print(game.observations((0, 5), h=16, w=16)[5])
