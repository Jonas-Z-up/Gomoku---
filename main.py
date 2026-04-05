import pygame
import sys
import numpy as np
import random
import math
from pygame.locals import *
from collections import defaultdict
import time

# Initialize the Pygame library for subsequent use
pygame.init()

# Game constants
SCREEN_WIDTH, SCREEN_HEIGHT = 880, 650
BOARD_SIZE = 15
GRID_SIZE = 40
BOARD_PADDING = 50
BOARD_WIDTH = GRID_SIZE * (BOARD_SIZE - 1)
PIECE_RADIUS = 18
ANIMATION_SPEED = 0.2

# Color definitions for the game
BACKGROUND = (240, 217, 181)
BOARD_COLOR = (220, 179, 92)
LINE_COLOR = (0, 0, 0)
BLACK_PIECE = (45, 45, 45)
WHITE_PIECE = (255, 255, 255)
HIGHLIGHT = (255, 0, 0, 100)
BUTTON_COLOR = (101, 67, 33)
BUTTON_HOVER = (139, 90, 43)
BUTTON_SELECTED = (80, 50, 20)
TEXT_COLOR = (255, 255, 255)
HINT_COLOR = (0, 128, 0, 100)

# Create the game window and set its size and title
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("五子棋")

# Load different fonts for displaying text in the game
title_font = pygame.font.SysFont("simhei", 48, bold=True)  # Title font
button_font = pygame.font.SysFont("simhei", 32)  # Button font
info_font = pygame.font.SysFont("simhei", 24)  # Information font
small_font = pygame.font.SysFont("simhei", 18)  # Small font


# Smart Monte Carlo Tree Search class for AI decision-making
class SmartMCTS:
    def __init__(self, board_size=13, c_param=1.4, simulations=300, max_depth=40):
        # Size of the chessboard
        self.board_size = board_size
        # Parameter in UCB algorithm for balancing exploration and exploitation
        self.c_param = c_param
        # Number of simulations
        self.simulations = simulations
        # Maximum depth of simulation
        self.max_depth = max_depth
        # Accumulated rewards for each action
        self.Q = defaultdict(float)
        # Number of visits for each action
        self.N = defaultdict(int)
        # Child nodes for each state
        self.children = dict()
        # Cache for chess patterns to avoid repeated calculations
        self.pattern_cache = {}

        # Offensive weights, especially sensitive to open fours and open threes
        self.offensive_weights = {
            "FIVE": 1000000,  # Five in a row (immediate win)
            "OPEN_FOUR": 800000,  # Open four (immediate threat), significantly increase weight
            "HALF_OPEN_FOUR": 15000,  # Half-open four
            "DOUBLE_THREE": 50000,  # Double three (key offensive move)
            "OPEN_THREE": 6000,  # Open three (increase weight)
            "HALF_OPEN_THREE": 1000,  # Half-open three
            "OPEN_TWO": 300,  # Open two
            "HALF_OPEN_TWO": 100,  # Half-open two
            "WIN_IN_ONE": 1000000  # Candidate point to form five in a row directly, full score
        }

        # Defensive weights adjustment
        self.defensive_weights = {
            "FIVE": 1000000,
            "OPEN_FOUR": 350000,  # Defend against open four (high priority)
            "HALF_OPEN_FOUR": 12000,
            "DOUBLE_THREE": 25000,  # Defend against opponent's double three
            "OPEN_THREE": 6000,  # Opponent's open three threat, increase weight
            "HALF_OPEN_THREE": 600,
            "OPEN_TWO": 100,
            "HALF_OPEN_TWO": 30
        }

    def get_legal_actions(self, board):
        """Get all legal moves (smart selection)"""
        if np.sum(board) == 0:
            center = self.board_size // 2
            return [(center, center)]

        actions = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != 0:
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            nr = r + dr
                            nc = c + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == 0:
                                actions.add((nr, nc))

        if not actions:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if board[r, c] == 0:
                        actions.add((r, c))

        return list(actions)

    def check_win(self, board, r, c):
        player = board[r, c]
        if player == 0:
            return False

        # Check directions: horizontal, vertical, two diagonals
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # Current position

            # Forward check
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc

            # Backward check
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 5:
                return True

        return False

    def simulate_move(self, board, player, r, c):
        """Simulate placing a piece at (r, c)"""
        new_board = board.copy()
        new_board[r, c] = player
        win = self.check_win(new_board, r, c)
        return new_board, win

    def evaluate_position(self, board, player):
        """Evaluate the value of the chessboard position (integrate extended open threes and center control)"""
        score = 0
        opponent = 3 - player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        center = self.board_size // 2

        # Build threat map: record directions of own open threes
        threat_map = np.zeros_like(board)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] == player:
                    for dr, dc in directions:
                        pattern = self.check_pattern(board, r, c, dr, dc, player)
                        if pattern == "OPEN_THREE":
                            for step in range(1, 3):
                                for sign in [-1, 1]:
                                    nr = r + dr * step * sign
                                    nc = c + dc * step * sign
                                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == 0:
                                        threat_map[nr, nc] += 1

        # Evaluate empty points
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != 0:
                    continue

                off_score = self.evaluate_single_point(board, r, c, player, offensive=True)
                def_score = self.evaluate_single_point(board, r, c, opponent, offensive=False)

                # Center weighting function (lower as the distance increases)
                center_weight = 1.0 - (abs(r - center) ** 1.2 + abs(c - center) ** 1.2) / (self.board_size ** 1.5)
                center_weight = max(0.3, center_weight)  # Limit the minimum value

                # Extra bonus: if it is an extended direction of an open three, add a certain bonus
                combo_bonus = threat_map[r, c] * self.offensive_weights["OPEN_THREE"] * 0.3

                total = (off_score + def_score + combo_bonus) * center_weight
                score = max(score, total)

        return score

    def evaluate_single_point(self, board, r, c, player, offensive=True):
        """Evaluate the value of a single position (enhanced key pattern detection + combination recognition)"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        weights = self.offensive_weights if offensive else self.defensive_weights

        # Temporarily place a piece for evaluation
        board[r, c] = player

        pattern_counts = defaultdict(int)
        patterns = []

        for dr, dc in directions:
            pattern = self.check_pattern(board, r, c, dr, dc, player)
            patterns.append(pattern)
            pattern_counts[pattern] += 1

            # If an open four appears, immediately return a high value
            if pattern == "OPEN_FOUR":
                board[r, c] = 0
                return weights["OPEN_FOUR"] * 2

            score += weights.get(pattern, 0)

        # Combination detection logic
        if pattern_counts["OPEN_THREE"] >= 2:
            score += weights["DOUBLE_THREE"] * 2
        elif pattern_counts["OPEN_THREE"] >= 1 and pattern_counts["HALF_OPEN_FOUR"] >= 1:
            score += weights["OPEN_THREE"] * 1.5

        # Open three + open two
        if pattern_counts["OPEN_THREE"] >= 1 and pattern_counts["OPEN_TWO"] >= 1:
            score += weights["OPEN_THREE"] * 0.8 + weights["OPEN_TWO"] * 0.5

        # Double half-open four recognition
        if pattern_counts["HALF_OPEN_FOUR"] >= 2:
            score += weights["HALF_OPEN_FOUR"] * 1.5

        # Symmetric attack point recognition
        if len(set(patterns)) == 1 and patterns[0] in ["OPEN_THREE", "OPEN_TWO"]:
            score += weights[patterns[0]] * 0.5

        # Combination threat recognition
        if pattern_counts["OPEN_FOUR"] >= 1 and pattern_counts["OPEN_THREE"] >= 1:
            score += weights["OPEN_FOUR"] * 0.8 + weights["OPEN_THREE"] * 0.8

        # Double open three
        if pattern_counts["OPEN_THREE"] >= 2:
            score += weights["DOUBLE_THREE"] * 1.2

        # Open three + half-open four
        if pattern_counts["OPEN_THREE"] >= 1 and pattern_counts["HALF_OPEN_FOUR"] >= 1:
            score += weights["OPEN_THREE"] * 0.5 + weights["HALF_OPEN_FOUR"] * 0.5

        # Five in a row bonus
        if self.check_win(board, r, c):
            score += weights.get("WIN_IN_ONE", 1000000)

        board[r, c] = 0
        return score

    def check_pattern(self, board, r, c, dr, dc, player):
        """Check the chess pattern in a specific direction"""
        # Get the pattern feature
        key = (r, c, dr, dc, player)
        if key in self.pattern_cache:
            return self.pattern_cache[key]

        # Calculate the number of consecutive pieces in the positive direction
        count = 1
        # Positive direction
        nr, nc = r + dr, c + dc
        while 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
            count += 1
            nr += dr
            nc += dc

        # Check if the positive direction is blocked
        forward_blocked = not (0 <= nr < self.board_size and 0 <= nc < self.board_size) or board[nr, nc] != 0

        # Negative direction
        nr, nc = r - dr, c - dc
        while 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
            count += 1
            nr -= dr
            nc -= dc

        # Check if the negative direction is blocked
        backward_blocked = not (0 <= nr < self.board_size and 0 <= nc < self.board_size) or board[nr, nc] != 0

        # Determine the pattern
        pattern = "NONE"

        if count >= 5:
            pattern = "FIVE"
        elif count == 4:
            if not forward_blocked and not backward_blocked:
                pattern = "OPEN_FOUR"
            elif not forward_blocked or not backward_blocked:
                pattern = "HALF_OPEN_FOUR"
        elif count == 3:
            if not forward_blocked and not backward_blocked:
                pattern = "OPEN_THREE"
            elif not forward_blocked or not backward_blocked:
                pattern = "HALF_OPEN_THREE"
        elif count == 2:
            if not forward_blocked and not backward_blocked:
                pattern = "OPEN_TWO"
            elif not forward_blocked or not backward_blocked:
                pattern = "HALF_OPEN_TWO"
        # Special check for real open three
        if count == 3 and not forward_blocked and not backward_blocked:
            if self.is_real_open_three(board, r, c, dr, dc, player):
                pattern = "OPEN_THREE"

        if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
            return "NONE"
        # Cache the result
        self.pattern_cache[key] = pattern
        return pattern

    def is_real_open_three(self, board, r, c, dr, dc, player):
        """Check if it is a real open three (at least two consecutive empty spaces on both sides)"""
        # Forward check
        open_forward = True
        nr, nc = r + dr, c + dc
        for _ in range(2):
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size) or board[nr, nc] != 0:
                open_forward = False
                break
            nr += dr
            nc += dc

        # Backward check
        open_backward = True
        nr, nc = r - dr, c - dc
        for _ in range(2):
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size) or board[nr, nc] != 0:
                open_backward = False
                break
            nr -= dr
            nc -= dc

        return open_forward and open_backward

    def ucb(self, state, action):
        """Calculate the UCB value"""
        state_key = state.tobytes()
        action_key = (action[0], action[1])

        if self.N[(state_key, action_key)] == 0:
            return float('inf')  # Unexplored nodes are explored first

        # Calculate UCB value: Q/N + c * sqrt(ln(parent_visits) / N)
        parent_visits = self.N[state_key]
        return (self.Q[(state_key, action_key)] / self.N[(state_key, action_key)]) + \
            self.c_param * math.sqrt(math.log(parent_visits) / self.N[(state_key, action_key)])

    def rollout(self, board, player):
        """Heuristic random simulation (enhanced offensive awareness)"""
        current_board = board.copy()
        current_player = player
        depth = 0

        # Try to search for offensive opportunities in the first few steps
        max_offensive_depth = min(5, self.max_depth)

        while depth < self.max_depth:
            actions = self.get_legal_actions(current_board)
            if not actions:
                return 0.5  # Draw

            # Prioritize offensive moves in the first few steps
            if depth < max_offensive_depth:
                best_score = -1
                best_actions = []
                for r, c in actions:
                    # Quick check for win
                    current_board[r, c] = current_player
                    if self.check_win(current_board, r, c):
                        current_board[r, c] = 0
                        return 1.0 if current_player == player else 0.0
                    current_board[r, c] = 0

                    # Evaluate offensive value
                    off_score = self.evaluate_single_point(current_board, r, c, current_player, True)
                    if off_score > best_score:
                        best_score = off_score
                        best_actions = [(r, c)]
                    elif off_score == best_score:
                        best_actions.append((r, c))

                # Select the best offensive move
                if best_actions:
                    r, c = random.choice(best_actions)
                else:
                    r, c = random.choice(actions)
            else:
                # Evaluate the value of all actions
                action_scores = []
                for r, c in actions:
                    # Quick check for win
                    current_board[r, c] = current_player
                    if self.check_win(current_board, r, c):
                        current_board[r, c] = 0
                        return 1.0 if current_player == player else 0.0
                    current_board[r, c] = 0

                    # Position evaluation
                    score = self.evaluate_single_point(current_board, r, c, current_player, True)
                    action_scores.append(score)

                # Select an action based on the value
                total_score = sum(action_scores)
                if total_score > 0:
                    probabilities = [score / total_score for score in action_scores]
                    r, c = random.choices(actions, weights=probabilities, k=1)[0]
                else:
                    r, c = random.choice(actions)

            # Execute the action
            current_board[r, c] = current_player

            # Check for win
            if depth == 0:
                for r, c in self.get_legal_actions(current_board):
                    current_board[r, c] = current_player
                    if self.check_win(current_board, r, c):
                        current_board[r, c] = 0
                        return 1.0 if current_player == player else 0.0
                    current_board[r, c] = 0

            # Switch players
            current_player = 3 - current_player
            depth += 1

        # Evaluate the position value if no winner is determined after reaching the maximum depth
        position_score = self.evaluate_position(current_board, player)
        max_score = self.offensive_weights["OPEN_FOUR"] * 2  # Normalization factor

        # Convert the position value to a win rate estimate
        return 0.4 + 0.2 * (position_score / max_score)  # In the range of 0.4 - 0.6

    def get_best_action(self, board, player):
        """Get the best action in the current state (including opening book, heat map, and enhanced strategy)"""
        self.debug_info = []  # Initialize debug information list

        actions = self.get_legal_actions(board)
        opponent = 3 - player
        total_moves = np.count_nonzero(board)
        center = self.board_size // 2

        self.debug_info.append(f"当前落子数: {total_moves}")
        self.debug_info.append(f"AI颜色: {'黑棋' if player == 1 else '白棋'}")
        self.debug_info.append(f"候选点数: {len(actions)}")

        print("\n[AI] 当前总落子数:", total_moves)
        print("[AI] 当前AI颜色:", "黑棋" if player == 1 else "白棋")
        print("[AI] 生成候选动作数:", len(actions))

        # [1] Opening book (first 5 moves)
        opening_book = {
            0: [(center, center)],
            1: [
                (center - 1, center - 1), (center - 1, center + 1),
                (center + 1, center - 1), (center + 1, center + 1),
                (center - 1, center), (center + 1, center),
                (center, center - 1), (center, center + 1)
            ],
            2: [
                (center - 2, center), (center + 2, center),
                (center, center - 2), (center, center + 2)
            ],
            3: [
                (center - 1, center + 2), (center + 1, center - 2),
                (center - 2, center - 1), (center + 2, center + 1)
            ],
            4: [
                (center - 2, center - 2), (center + 2, center + 2),
                (center - 3, center), (center, center + 3)
            ]
        }

        if total_moves in opening_book:
            legal_moves = [
                move for move in opening_book[total_moves]
                if 0 <= move[0] < self.board_size and 0 <= move[1] < self.board_size and board[move[0], move[1]] == 0
            ]
            if legal_moves:
                random_message = random.choice(["下棋啦嘿嘿嘿! (=•ω•=)", "下棋啦嘿嘿嘿! o(*^＠^*)o", "开局! <(￣︶￣)>"])
                self.debug_info.append(random_message)
                return random.choice(legal_moves)

        immediate_win = []
        critical_attacks = []
        critical_defenses = []

        # [2] Heat map
        heat_map = np.zeros((self.board_size, self.board_size))
        for r, c in actions:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        if board[nr, nc] != 0:
                            heat_map[r, c] += 1

        # [3] Heuristic analysis
        scored_actions = []
        for r, c in actions:
            if self.check_win_after_move(board, r, c, player):
                self.debug_info.append(f"直接胜点: ({r},{c})，赢啦哈哈哈!")
                self.debug_info.append("(≧ω≦)")
                return (r, c)

            off_score = self.evaluate_single_point(board, r, c, player, offensive=True)
            def_score = self.evaluate_single_point(board, r, c, opponent, offensive=False)

            if off_score >= self.offensive_weights["HALF_OPEN_FOUR"] * 1.2 and def_score < self.defensive_weights[
                "OPEN_THREE"]:
                critical_attacks.append((r, c))
            if def_score >= self.defensive_weights["OPEN_FOUR"] * 0.9:
                critical_defenses.append((r, c))
            elif def_score >= self.defensive_weights["HALF_OPEN_FOUR"] * 1.2:
                critical_defenses.append((r, c))

            score = off_score + def_score + heat_map[r, c] * 20
            scored_actions.append(((r, c), score))

            print(f"[AI评估] 点({r},{c}) 进攻:{off_score:.1f} 防守:{def_score:.1f} 总分:{score:.1f}")
            if len(scored_actions) <= 5:
                self.debug_info.append(f"评估({r},{c}) 进:{int(off_score)},防:{int(def_score)}")

            if off_score >= self.offensive_weights["OPEN_FOUR"] * 0.9:
                immediate_win.append((r, c))
            elif off_score >= self.offensive_weights["DOUBLE_THREE"] * 0.8:
                critical_attacks.append((r, c))

            new_board_opp, win_opp = self.simulate_move(board, opponent, r, c)
            if win_opp:
                self.debug_info.append(f"预判防守: ({r},{c}) 阻止对方胜利")
                random_message = random.choice(["想赢我没那么容易! ㄟ( ▔, ▔ )ㄏ", "好险! o(￣ヘ￣o＃)"])
                self.debug_info.append(random_message)
                return (r, c)

            if def_score >= self.defensive_weights["OPEN_FOUR"] * 0.9:
                critical_defenses.append((r, c))
            elif def_score >= self.defensive_weights["DOUBLE_THREE"] * 0.7:
                critical_defenses.append((r, c))

        if immediate_win:
            self.debug_info.append(f"立即胜点: {immediate_win},要赢啦!")
            self.debug_info.append("︿(￣︶￣)︿")
            print("[AI] 发现直接获胜点:", immediate_win)
            return random.choice(immediate_win)
        if critical_attacks:
            self.debug_info.append(f"关键进攻点: {critical_attacks},看到胜利曙光啦!")
            self.debug_info.append("^_^")
            print("[AI] 检测到关键进攻点:", critical_attacks)
            return random.choice(critical_attacks)
        if critical_defenses:
            self.debug_info.append(f"关键防守点: {critical_defenses}")
            random_message = random.choice(["防守防守!!! (°ロ°) !", "不能让你赢!!! w(°Д°)w"])
            self.debug_info.append(random_message)
            print("[AI] 检测到关键防守点:", critical_defenses)
            return random.choice(critical_defenses)

        # [4] Monte Carlo search preparation
        scored_actions.sort(key=lambda x: -x[1])
        top_actions = [a for a, _ in scored_actions[:min(12, len(scored_actions))]]
        self.children[board.tobytes()] = top_actions

        for _ in range(self.simulations):
            best_action = None
            best_ucb = -float("inf")
            for action in top_actions:
                ucb_val = self.ucb(board, action)
                if ucb_val > best_ucb:
                    best_ucb = ucb_val
                    best_action = action

            if best_action is None:
                best_action = random.choice(top_actions)

            r, c = best_action
            new_board, win = self.simulate_move(board, player, r, c)
            result = 1.0 if win else self.rollout(new_board, 3 - player)

            action_key = (r, c)
            state_key = board.tobytes()
            self.N[(state_key, action_key)] += 1
            self.Q[(state_key, action_key)] += result
            self.N[state_key] = self.N.get(state_key, 0) + 1

        best_action = max(top_actions, key=lambda a: self.N.get((board.tobytes(), a), 0))
        self.debug_info.append(f"最终选择: {best_action},此棋何解？")
        self.debug_info.append("o(￣▽￣)ｄ")
        print("[AI] 最终选择动作:", best_action)
        return best_action

    def check_win_after_move(self, board, r, c, player):
        """Separate function for quick win check in get_best_action"""
        board[r, c] = player
        result = self.check_win(board, r, c)
        board[r, c] = 0
        return result


# Button class for creating buttons in the game
class Button:
    def __init__(self, x, y, width, height, text):
        # Create a rectangular area for the button
        self.rect = pygame.Rect(x, y, width, height)
        # Text displayed on the button
        self.text = text
        # Flag indicating whether the mouse is hovering over the button
        self.hovered = False
        # Flag indicating whether the button is selected
        self.selected = False

    def draw(self, surface):
        # Select the color based on the button's state
        if self.selected:
            color = BUTTON_SELECTED
        else:
            color = BUTTON_HOVER if self.hovered else BUTTON_COLOR

        # Draw the button's background rectangle
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        # Draw the button's border
        pygame.draw.rect(surface, (60, 40, 20), self.rect, 3, border_radius=10)

        # Render the text on the button
        text_surf = button_font.render(self.text, True, TEXT_COLOR)
        # Get the rectangular area of the text and center it
        text_rect = text_surf.get_rect(center=self.rect.center)
        # Draw the text on the button
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        # Check if the mouse position is within the button area and update the hover flag
        self.hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos, event):
        # Check if the button is clicked
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

def render_multiline_text(screen, text, font, color, x, y, max_width, line_spacing=4):
    """
    Automatically wrap the text that exceeds the width and draw it
    """
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())

    for i, line in enumerate(lines):
        text_surface = font.render(line, True, color)
        screen.blit(text_surface, (x, y + i * (font.get_linesize() + line_spacing)))
    return len(lines)  # Return the number of drawn lines

# Game class for managing the game state and logic
class Game:
    def __init__(self):
        # Initial game state: menu, in-game, game over
        self.state = "menu"
        # Initialize the chessboard as a numpy array
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        # Current player: 1 for player (black), 2 for AI (white)
        self.current_player = 1
        # Flag indicating whether the game is over
        self.game_over = False
        # Winner: 0 for draw
        self.winner = 0
        # Last move position
        self.last_move = None
        # Progress of the chess piece animation
        self.animation_progress = 0
        # Flag indicating whether the chess piece is animating
        self.animating = False
        # Information of the animating chess piece
        self.animate_piece = None
        # Start game button
        self.start_button = Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 60, "开始游戏")
        self.restart_button = Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 100, 200, 60, "重新开始")
        self.first_hand_button = Button(SCREEN_WIDTH // 2 - 220, SCREEN_HEIGHT // 2 - 50, 200, 60, "先手(黑棋)")
        self.second_hand_button = Button(SCREEN_WIDTH // 2 + 20, SCREEN_HEIGHT // 2 - 50, 200, 60, "后手(白棋)")
        # First hand is selected by default
        self.first_hand_button.selected = True

        # List of highlighted positions
        self.highlighted_positions = []
        # Flag indicating whether the AI is thinking
        self.ai_thinking = False
        # Player's color: 1 for black, 2 for white
        self.player_color = 1
        # Record the history of each move
        self.move_history = []
        # Undo button
        self.undo_button = Button(BOARD_PADDING + BOARD_WIDTH + 20, BOARD_PADDING + 100, 120, 40, "悔棋")
        # Red circle mark for the last move
        self.last_move_player = None
        self.last_move_ai = None
        # Replay button
        self.replay_button = Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 160, 200, 60, "胜利回放")

        self.replaying = False  # Whether in replay mode
        self.replay_index = 0  # Current replay step
        self.replay_moves = []  # List of replay steps
        self.replay_timer = 0  # Timer for replay interval control
        self.show_menu_overlay = True  # Whether to show the menu overlay
        self.menu_close_button = Button(SCREEN_WIDTH - 50, 20, 30, 30, "×")

        self.ai = None  # Initialize as None to avoid undefined error

    def reset(self):
        # Reset the chessboard
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        # Black starts first
        self.current_player = 1
        # Game is not over
        self.game_over = False
        # No winner
        self.winner = 0
        # Clear the last move position
        self.last_move = None
        # Clear the highlighted positions
        self.highlighted_positions = []

        # Set the player's color based on the first or second hand selection
        if self.player_color == 2:  # If the player chooses the second hand
            # AI makes the first move
            self.ai_thinking = True
        self.move_history = []  # Clear the move history when resetting
        # Red circle mark for the last move
        self.last_move_player = None
        self.last_move_ai = None
        # Menu overlay
        self.show_menu_overlay = True

    def undo_move(self):
        """Undo the last move - remove the player's and AI's last moves"""
        # At least 2 moves are required to undo (player's move and AI's move)
        if len(self.move_history) < 2 or self.game_over:
            return

        # Get the last two moves
        last_player_move = self.move_history[-1]
        last_ai_move = self.move_history[-2]

        # Remove the pieces from the chessboard
        self.board[last_player_move[0]][last_player_move[1]] = 0
        self.board[last_ai_move[0]][last_ai_move[1]] = 0

        # Update the move history (remove the last two moves)
        self.move_history = self.move_history[:-2]

        # Update the last move position
        if self.move_history:
            self.last_move = (self.move_history[-1][0], self.move_history[-1][1])
        else:
            self.last_move = None

        # Reset the game state
        self.game_over = False
        self.winner = 0
        self.highlighted_positions = []

        # Ensure the current player is the player (after undo, it's the player's turn again)
        self.current_player = self.player_color

        # Stop any ongoing AI thinking
        self.ai_thinking = False
        self.animating = False

        self.last_move_player = None
        self.last_move_ai = None

    def make_move(self, row, col):
        # Check if it's the player's turn
        if (self.current_player == 1 and self.player_color == 1) or \
                (self.current_player == 2 and self.player_color == 2):

            if self.board[row][col] == 0 and not self.game_over:
                # Place a piece at the specified position
                self.board[row][col] = self.current_player
                # Record the last move position
                self.last_move = (row, col)
                # Set the information of the animating chess piece
                self.animate_piece = (row, col, self.current_player)

                if self.current_player == self.player_color:
                    self.last_move_player = (row, col)
                else:
                    self.last_move_ai = (row, col)

                # Start the chess piece animation
                self.animating = True
                # Initialize the animation progress
                self.animation_progress = 0

                # Check if the game is over
                if self.check_win(row, col):
                    self.game_over = True
                    self.winner = self.current_player
                    self.state = "game_over"
                    # Find the winning positions
                    self.find_winning_positions(row, col)
                elif self.is_board_full():
                    self.game_over = True
                    self.state = "game_over"
                else:
                    # Switch players
                    self.current_player = 3 - self.current_player

                    # If it's the AI's turn, set the AI thinking flag
                    if (self.current_player == 1 and self.player_color == 2) or \
                            (self.current_player == 2 and self.player_color == 1):
                        # Start thinking after the animation ends
                        pass
        self.move_history.append((row, col, self.current_player))

    def process_ai_move(self):
        if self.ai_thinking:
            # Process the AI's move
            self.ai_move()
            self.ai_thinking = False

    def ai_move(self):
        if self.game_over:
            return

        # Dynamically adjust the number of simulations based on the game stage
        pieces_count = np.count_nonzero(self.board)
        if pieces_count < 20:  # Opening stage
            simulations = 300
        elif pieces_count < 60:  # Mid-game stage
            simulations = 400
        else:  # End-game stage
            simulations = 300

        # Create a SmartMCTS instance
        self.ai = SmartMCTS(board_size=BOARD_SIZE, simulations=simulations)

        # Get the best move
        best_move = self.ai.get_best_action(self.board, self.current_player)

        if best_move:
            r, c = best_move
            # Place a piece at the best position
            self.board[r][c] = self.current_player
            # Record the last move position
            self.last_move = (r, c)
            # Set the information of the animating chess piece
            self.animate_piece = (r, c, self.current_player)

            if self.current_player == self.player_color:
                self.last_move_player = (r, c)
            else:
                self.last_move_ai = (r, c)

            # Start the chess piece animation
            self.animating = True
            # Initialize the animation progress
            self.animation_progress = 0

            # Check if the game is over
            if self.check_win(r, c):
                self.game_over = True
                self.winner = self.current_player
                self.state = "game_over"
                # Find the winning positions
                self.find_winning_positions(r, c)
            elif self.is_board_full():
                self.game_over = True
                self.state = "game_over"
            else:
                # Switch back to the player
                self.current_player = 3 - self.current_player
        self.move_history.append((r, c, self.current_player))
    def check_win(self, row, col):
        # Check if the player at the given position wins
        player = self.board[row][col]
        if player == 0:
            return False

        # Directions: horizontal, vertical, two diagonals
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # Current position

            # Check forward
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Check backward
            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False


    def find_winning_positions(self, row, col):
        # Find the winning positions starting from the given position
        player = self.board[row][col]
        if player == 0:
            return []

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        winning_positions = []

        for dr, dc in directions:
            positions = [(row, col)]

            # Check forward
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                positions.append((r, c))
                r += dr
                c += dc

            # Check backward
            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                positions.append((r, c))
                r -= dr
                c -= dc

            if len(positions) >= 5:
                winning_positions = positions
                break

        self.highlighted_positions = winning_positions
        return winning_positions


    def is_board_full(self):
        # Check if the board is full
        return np.count_nonzero(self.board) == BOARD_SIZE * BOARD_SIZE

    def draw(self, screen):
        # Fill the background color
        screen.fill(BACKGROUND)

        if self.state == "menu":
            # Draw the menu interface
            self.draw_menu(screen)
        elif self.state in ["playing", "game_over", "replay"]:
            # Draw the board
            self.draw_board(screen)
            # Draw the pieces
            self.draw_pieces(screen)

            if self.state == "game_over":
                # Draw the game over interface
                self.draw_game_over(screen)
            if self.state == "playing":
                self.undo_button.draw(screen)


    def draw_menu(self, screen):
        title = title_font.render("五子棋", True, (70, 45, 20))
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, SCREEN_HEIGHT // 4 - 50))
        # Draw the board background
        board_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 4, 300, 300)
        pygame.draw.rect(screen, BOARD_COLOR, board_rect)
        pygame.draw.rect(screen, (60, 40, 20), board_rect, 4)

        # Draw the grid
        for i in range(15):
            # Horizontal lines
            pygame.draw.line(screen, LINE_COLOR,
                             (board_rect.left + 20, board_rect.top + 20 + i * 20),
                             (board_rect.right - 20, board_rect.top + 20 + i * 20), 2)
            # Vertical lines
            pygame.draw.line(screen, LINE_COLOR,
                             (board_rect.left + 20 + i * 20, board_rect.top + 20),
                             (board_rect.left + 20 + i * 20, board_rect.bottom - 20), 2)

        # Draw example pieces
        pygame.draw.circle(screen, BLACK_PIECE,
                           (board_rect.centerx - 30, board_rect.centery - 30), 10)
        pygame.draw.circle(screen, WHITE_PIECE,
                           (board_rect.centerx + 30, board_rect.centery + 30), 10)

        # Draw the first hand and second hand selection buttons
        self.first_hand_button.draw(screen)
        self.second_hand_button.draw(screen)
        self.start_button.draw(screen)
        rules = [
            "游戏规则:",
            "1. 选择先后手开始游戏",
            "2. 在棋盘交叉点上落子",
            "3. 先形成五子连线者获胜"
        ]

        for i, rule in enumerate(rules):
            text = small_font.render(rule, True, (70, 45, 20))
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 + 150 + i * 25))


    def draw_board(self, screen):
        # Draw the board background
        board_rect = pygame.Rect(BOARD_PADDING, BOARD_PADDING, BOARD_WIDTH, BOARD_WIDTH)
        pygame.draw.rect(screen, BOARD_COLOR, board_rect)
        pygame.draw.rect(screen, (101, 67, 33), board_rect, 4)

        # Draw the grid lines
        for i in range(BOARD_SIZE):
            # Horizontal lines
            pygame.draw.line(screen, LINE_COLOR,
                             (BOARD_PADDING, BOARD_PADDING + i * GRID_SIZE),
                             (BOARD_PADDING + BOARD_WIDTH, BOARD_PADDING + i * GRID_SIZE), 2)
            # Vertical lines
            pygame.draw.line(screen, LINE_COLOR,
                             (BOARD_PADDING + i * GRID_SIZE, BOARD_PADDING),
                             (BOARD_PADDING + i * GRID_SIZE, BOARD_PADDING + BOARD_WIDTH), 2)

        # Draw the star points
        star_points = [3, BOARD_SIZE // 2, BOARD_SIZE - 4]
        for r in star_points:
            for c in star_points:
                pygame.draw.circle(screen, LINE_COLOR,
                                   (BOARD_PADDING + c * GRID_SIZE, BOARD_PADDING + r * GRID_SIZE), 5)

        # Draw the current player information
        if self.player_color == 1:
            player_text = "当前回合: " + ("玩家(黑棋)" if self.current_player == 1 else "AI(白棋)")
        else:
            player_text = "当前回合: " + ("AI(黑棋)" if self.current_player == 1 else "玩家(白棋)")
        text_surf = info_font.render(player_text, True, (70, 45, 20))
        screen.blit(text_surf, (BOARD_PADDING + BOARD_WIDTH + 20, BOARD_PADDING + 30))

        # Draw the AI thinking information
        if self.ai_thinking:
            thinking_text = "AI正在思考……"
            thinking_surf = info_font.render(thinking_text, True, (70, 45, 20))
            screen.blit(thinking_surf, (BOARD_PADDING + BOARD_WIDTH + 20, BOARD_PADDING + 60))
        # Display AI thinking information
        if hasattr(self, "ai") and hasattr(self.ai, "debug_info"):
            font = pygame.font.SysFont("simhei", 20)
            # Starting x-coordinate
            start_x = SCREEN_WIDTH - 250
            # Starting y-coordinate
            start_y = 200
            line_spacing = 24

            screen.blit(font.render("AI 思考过程：", True, (0, 0, 0)), (start_x, start_y))
            # Display thinking content
            line_y = start_y + 27
            max_width = 180  # Maximum width for text wrapping

            for line in self.ai.debug_info[-6:]:
                line_count = render_multiline_text(screen, line, font, (50, 50, 50), start_x, line_y, max_width)
                line_y += line_count * (font.get_linesize() + 4)


    def draw_pieces(self, screen):
        # Draw all pieces
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != 0 and (
                        not self.animating or (r, c) != (self.animate_piece[0], self.animate_piece[1])):
                    center_x = BOARD_PADDING + c * GRID_SIZE
                    center_y = BOARD_PADDING + r * GRID_SIZE

                    color = BLACK_PIECE if self.board[r][c] == 1 else WHITE_PIECE
                    pygame.draw.circle(screen, color, (center_x, center_y), PIECE_RADIUS)
                    pygame.draw.circle(screen, LINE_COLOR, (center_x, center_y), PIECE_RADIUS, 1)

                    # Add gloss effect for white pieces
                    if self.board[r][c] == 2:
                        pygame.draw.circle(screen, (220, 220, 220),
                                           (center_x - PIECE_RADIUS // 3, center_y - PIECE_RADIUS // 3),
                                           PIECE_RADIUS // 3)

        # Draw the animating piece
        if self.animating:
            r, c, player = self.animate_piece
            center_x = BOARD_PADDING + c * GRID_SIZE
            center_y = BOARD_PADDING + r * GRID_SIZE

            # Calculate animation progress
            progress = min(1.0, self.animation_progress)
            radius = int(PIECE_RADIUS * progress)

            if radius > 0:
                color = BLACK_PIECE if player == 1 else WHITE_PIECE
                pygame.draw.circle(screen, color, (center_x, center_y), radius)
                pygame.draw.circle(screen, LINE_COLOR, (center_x, center_y), radius, 1)

                # Add gloss effect for white pieces
                if player == 2:
                    pygame.draw.circle(screen, (220, 220, 220),
                                       (center_x - radius // 3, center_y - radius // 3),
                                       radius // 3)

            # Update animation progress
            self.animation_progress += ANIMATION_SPEED
            if self.animation_progress > 1.5:
                self.animating = False
                self.animation_progress = 0
                # Check if AI needs to move after animation
                if (self.current_player == 1 and self.player_color == 2) or \
                        (self.current_player == 2 and self.player_color == 1):
                    self.ai_thinking = True
                    self.process_ai_move()

        # Highlight the winning pieces
        for pos in self.highlighted_positions:
            r, c = pos
            center_x = BOARD_PADDING + c * GRID_SIZE
            center_y = BOARD_PADDING + r * GRID_SIZE

            pygame.draw.circle(screen, HIGHLIGHT, (center_x, center_y), PIECE_RADIUS - 3, 3)

        # Show the last move of the current player
        if self.current_player == self.player_color:
            # Player's turn, show AI's last move
            if self.last_move_ai:
                r, c = self.last_move_ai
                center_x = BOARD_PADDING + c * GRID_SIZE
                center_y = BOARD_PADDING + r * GRID_SIZE
                pygame.draw.circle(screen, (0, 0, 255), (center_x, center_y), PIECE_RADIUS - 3, 3)  # Red circle
                pygame.draw.circle(screen, (0, 0, 255), (center_x, center_y), 4)  # Red dot
        else:
            # AI's turn, show player's last move
            if self.last_move_player:
                r, c = self.last_move_player
                center_x = BOARD_PADDING + c * GRID_SIZE
                center_y = BOARD_PADDING + r * GRID_SIZE
                pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), PIECE_RADIUS - 3, 3)  # Red circle
                pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), 4)  # Red dot
        # Highlight the replaying moves
        if self.state == "replay":
            for i in range(self.replay_index):
                if i < len(self.replay_moves):
                    r, c, _ = self.replay_moves[i]
                    center_x = BOARD_PADDING + c * GRID_SIZE
                    center_y = BOARD_PADDING + r * GRID_SIZE
                    pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), PIECE_RADIUS - 4, 2)


    def draw_game_over(self, screen):
        if not self.show_menu_overlay:
            return  # Don't draw menu if it's closed
        # Draw the semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))

        # Draw the result box
        result_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 100, 400, 200)
        pygame.draw.rect(screen, (240, 240, 240), result_rect, border_radius=15)
        pygame.draw.rect(screen, (101, 67, 33), result_rect, 4, border_radius=15)

        # Display the result text
        if self.winner == self.player_color:
            result_text = "恭喜！你赢了！"
            text_color = (50, 50, 50)
        elif self.winner == 3 - self.player_color:
            result_text = "AI获胜！"
            text_color = (50, 50, 50)
        else:
            result_text = "平局！"
            text_color = (50, 50, 50)

        text_surf = title_font.render(result_text, True, text_color)
        screen.blit(text_surf, (SCREEN_WIDTH // 2 - text_surf.get_width() // 2, SCREEN_HEIGHT // 2 - 70))

        # Draw the restart button
        self.restart_button.draw(screen)
        # Draw the replay button
        self.replay_button.draw(screen)
        # Draw the close button in the top right corner
        self.menu_close_button.draw(screen)


    def start_replay(self):
        print("[REVIEW] 回放开始！")
        self.replaying = True
        self.replay_index = 0
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.highlighted_positions = []
        self.last_move = None
        self.last_move_player = None
        self.last_move_ai = None

        N = 6  # Replay the last N moves
        total = len(self.move_history)

        if total >= N:
            self.fixed_moves = self.move_history[:total - N]
            self.replay_moves = self.move_history[total - N:]
        else:
            self.fixed_moves = []
            self.replay_moves = self.move_history[:]

        # Place pieces alternately from the beginning
        current_player = 1
        for i, (r, c, _) in enumerate(self.fixed_moves):
            self.board[r][c] = current_player
            current_player = 3 - current_player  # Alternate between 1 and 2

        self.state = "replay"
        self.replay_timer = time.time()


# Monte Carlo Tree Search Node Class
class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, parent_action=None):
        # Current state of the node
        self.state = state
        # Parent node
        self.parent = parent
        # Action from parent node to current node
        self.parent_action = parent_action
        # List of child nodes
        self.children = []
        # Number of visits to the current node
        self._number_of_visits = 0
        # Accumulated rewards for different results
        self._results = {1: 0, 2: 0, 0: 0}  # 0 represents a draw
        # List of untried actions
        self._untried_actions = None
        # Current player
        self.player = 1 if parent is None else 3 - parent.player

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            # Get legal actions in the current state
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        # Calculate the accumulated reward of the current node
        wins = self._results[1] if self.player == 1 else self._results[2]
        losses = self._results[2] if self.player == 1 else self._results[1]
        return wins - losses

    @property
    def n(self):
        # Get the number of visits to the current node
        return self._number_of_visits

    def expand(self):
        # Expand the node by choosing an untried action and creating a child node
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        # Check if the current node is a terminal node
        return self.state.is_game_over()

    def rollout(self):
        # Perform a simulation until the game ends
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def rollout_policy(self, possible_moves):
        # Randomly choose an action
        return random.choice(possible_moves)

    def backpropagate(self, result):
        # Backpropagate the result and update the node's visit count and accumulated reward
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        # Check if the node is fully expanded
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        # Select the best child node
        choices_weights = [
            (child.q / child.n) + c_param * math.sqrt((2 * math.log(self.n) / child.n))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def _tree_policy(self):
        # Tree search policy to select a node for expansion
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulations_number):
        # Perform simulations and select the best action
        for _ in range(simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child(c_param=0.0).parent_action

    # Create a game instance


game = Game()

# Main game loop
clock = pygame.time.Clock()
running = True

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Menu interface event handling
        if game.state == "menu":
            game.start_button.check_hover(mouse_pos)
            game.first_hand_button.check_hover(mouse_pos)
            game.second_hand_button.check_hover(mouse_pos)

            if game.start_button.is_clicked(mouse_pos, event):
                game.state = "playing"
                game.reset()
                if game.player_color == 2:  # If player chooses second hand, mark AI to think
                    game.ai_needs_to_think = True  # New flag
            elif game.first_hand_button.is_clicked(mouse_pos, event):
                # Choose first hand
                game.first_hand_button.selected = True
                game.second_hand_button.selected = False
                game.player_color = 1  # Player plays black
            elif game.second_hand_button.is_clicked(mouse_pos, event):
                # Choose second hand
                game.first_hand_button.selected = False
                game.second_hand_button.selected = True
                game.player_color = 2  # Player plays white

        # Game interface event handling
        elif game.state == "playing" and not game.animating:
            game.undo_button.check_hover(mouse_pos)  # Check hover state of undo button

            # Check undo button click
            if game.undo_button.is_clicked(mouse_pos, event):
                game.undo_move()
            # Check if it's the player's turn
            if ((game.current_player == 1 and game.player_color == 1) or
                    (game.current_player == 2 and game.player_color == 2)):

                if event.type == MOUSEBUTTONDOWN and event.button == 1:
                    # Check if the board is clicked
                    if (BOARD_PADDING <= mouse_pos[0] <= BOARD_PADDING + BOARD_WIDTH and
                            BOARD_PADDING <= mouse_pos[1] <= BOARD_PADDING + BOARD_WIDTH):

                        # Calculate the clicked position on the board
                        col = round((mouse_pos[0] - BOARD_PADDING) / GRID_SIZE)
                        row = round((mouse_pos[1] - BOARD_PADDING) / GRID_SIZE)

                        # Ensure the position is within the board
                        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                            game.make_move(row, col)

        # Game over interface event handling
        elif game.state == "game_over":
            game.menu_close_button.check_hover(mouse_pos)
            if game.menu_close_button.is_clicked(mouse_pos, event):
                game.show_menu_overlay = False  # Hide the menu overlay without changing state

            game.restart_button.check_hover(mouse_pos)
            if game.restart_button.is_clicked(mouse_pos, event):
                game.state = "playing"
                game.reset()
                if game.player_color == 2:  # If player chooses second hand, mark AI to think
                    game.ai_needs_to_think = True  # New flag
            # Replay button click handling
            game.replay_button.check_hover(mouse_pos)
            if game.replay_button.is_clicked(mouse_pos, event):
                game.start_replay()

    # Replay handling
    if game.replaying and game.state == "replay":
        now = time.time()
        if game.replay_index < len(game.replay_moves) and now - game.replay_timer > 0.6:
            r, c, _ = game.replay_moves[game.replay_index]

            step_number = len(game.fixed_moves) + game.replay_index
            current_player = 1 if step_number % 2 == 0 else 2  # Black starts first

            game.board[r][c] = current_player
            game.last_move = (r, c)

            if current_player == game.player_color:
                game.last_move_player = (r, c)
            else:
                game.last_move_ai = (r, c)

            game.replay_index += 1
            game.replay_timer = now
        elif game.replay_index >= len(game.replay_moves):
            game.replaying = False
            game.state = "game_over"

    # Draw the game
    game.draw(screen)

    # Update the screen
    pygame.display.flip()

    # Handle AI move
    if hasattr(game, 'ai_needs_to_think') and game.ai_needs_to_think and not game.animating:
        game.ai_thinking = True
        game.process_ai_move()
        game.ai_needs_to_think = False

    clock.tick(60)

pygame.quit()
sys.exit()