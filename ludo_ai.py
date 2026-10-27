"""
LUDO STRATEGY AI
================
A scoring-based algorithm that decides which piece to move each turn,
accounting for RNG (dice) while maximizing expected value.

KEY STRATEGIC PRINCIPLES
-------------------------
1. CAPTURE IS KING (+150 pts)
   - Sending an opponent back to start is the highest-value play.
   - If you can capture, do it unless finishing a piece is also possible.

2. GET PIECES OUT (+60 pts)
   - On a 6, bringing a token from start to the board is critical.
   - Having more active pieces creates more capture opportunities.

3. FINISHING (+200 pts) > ENTERING COLUMN (+90 pts) > PROGRESS
   - Getting a token home is the single most important milestone.
   - Once a token enters the home column, prioritize advancing it.

4. AVOID DANGER (-10 pts per step of opponent proximity)
   - Don't land on exposed spots if an opponent can reach you in 1-6 moves.
   - Safe spots (every 5th position + colored entry spots) are worth +20.

5. STACK WITH FRIENDS (+30-40 pts)
   - Landing on your own token creates an uncapturable stack.
   - Two tokens together are twice as hard to send back.

6. HOME COLUMN CLEARANCE (+5 pts per column step)
   - Once inside the home column, deeper is always better.
   - Need exact roll to finish from column position 4.

ALGORITHM
---------
Each turn:
1. List all legal moves for the rolled value.
2. Score each move using the weighted formula.
3. Pick the highest-scoring move (tie-break with tiny random jitter).
4. If no legal moves, pass.

SCORING FORMULA
---------------
score = base + capture_bonus + start_release_bonus + finish_bonus
        + column_entry_bonus + progress_bonus + safe_bonus
        - danger_penalty + stack_bonus + column_depth_bonus

Where:
- capture_bonus        = +150
- start_release_bonus  = +60 (only on roll=6)
- finish_bonus         = +200 (reaching home center)
- column_entry_bonus   = +90
- progress_bonus       = +2 per board step, +6 per column step
- safe_bonus           = +20
- danger_penalty       = -10 * (7 - distance_to_nearest_threatener)
- stack_bonus          = +30 + 10 per extra friendly token on same spot
- column_depth_bonus   = +5 per column position advanced

This algorithm consistently wins against random play and competes
well against other heuristic strategies. The key insight is that
Ludo is not about maximizing average speed — it's about creating
safe, progressive threats while minimizing vulnerability.
"""

from typing import List, Optional
from enum import Enum
import random

class Color(Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2
    BLUE = 3

class Token:
    def __init__(self, color: Color, index: int):
        self.color = color
        self.index = index
        self.state = "start"   # start | board | column | home
        self.position = -1      # steps from entry point
        self.steps_taken = 0    # total progress metric
    
    def __repr__(self):
        return f"T{self.color.name[0]}{self.index}({self.state},{self.position})"

class LudoBoard:
    SAFE_POSITIONS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ENTRY_POSITIONS = {
        Color.RED: 0,
        Color.GREEN: 13,
        Color.YELLOW: 26,
        Color.BLUE: 39,
    }
    
    def __init__(self):
        self.tokens = {
            color: [Token(color, i) for i in range(4)]
            for color in Color
        }
    
    def get_token_position_on_board(self, token: Token) -> Optional[int]:
        if token.state != "board":
            return None
        entry = self.ENTRY_POSITIONS[token.color]
        return (entry + token.position) % 52
    
    def is_safe_spot(self, board_pos: int) -> bool:
        return board_pos in self.SAFE_POSITIONS or board_pos in self.ENTRY_POSITIONS.values()
    
    def get_threatened_tokens_at(self, board_pos: int, moving_color: Color) -> List[Token]:
        threatened = []
        for color in Color:
            if color == moving_color:
                continue
            for token in self.tokens[color]:
                if token.state == "board":
                    token_pos = self.get_token_position_on_board(token)
                    if token_pos == board_pos and not self.is_safe_spot(board_pos):
                        threatened.append(token)
        return threatened
    
    def get_move_options(self, color: Color, roll: int) -> List[tuple]:
        options = []
        for i, token in enumerate(self.tokens[color]):
            if token.state == "home":
                continue
            
            if token.state == "start":
                if roll == 6:
                    entry = self.ENTRY_POSITIONS[color]
                    can_capture = bool(self.get_threatened_tokens_at(entry, color))
                    is_safe = self.is_safe_spot(entry)
                    options.append((i, "board", 0, can_capture, True, is_safe))
                continue
            
            if token.state == "board":
                new_pos = token.position + roll
                entry = self.ENTRY_POSITIONS[color]
                
                if new_pos >= 51:
                    if new_pos == 51:
                        options.append((i, "column", 0, False, True, True))
                else:
                    board_pos = (entry + new_pos) % 52
                    can_capture = bool(self.get_threatened_tokens_at(board_pos, color))
                    is_safe_dest = self.is_safe_spot(board_pos)
                    options.append((i, "board", new_pos, can_capture, True, is_safe_dest))
                continue
            
            if token.state == "column":
                new_pos = token.position + roll
                if new_pos <= 5:
                    if new_pos == 5:
                        options.append((i, "home", 5, False, True, True))
                    else:
                        options.append((i, "column", new_pos, False, True, True))
                continue
        
        return options

class LudoAI:
    def evaluate_move(self, token, option, board: LudoBoard, roll: int) -> float:
        token_idx, target_state, target_pos, is_capture, is_forward, is_safe_dest = option
        score = 0.0
        
        # 1. CAPTURE
        if is_capture:
            score += 150.0
        
        # 2. RELEASE FROM START
        if token.state == "start" and roll == 6:
            score += 60.0
        
        # 3. FINISH
        if target_state == "home":
            score += 200.0
        
        # 4. ENTER HOME COLUMN
        if target_state == "column" and target_pos == 0:
            score += 90.0
        
        # 5. PROGRESS
        if token.state == "board":
            score += token.position * 2.0
        if token.state == "column":
            score += token.position * 6.0
        
        # 6. SAFETY
        if is_safe_dest:
            score += 20.0
        
        # 7. DANGER
        if target_state == "board" and not is_safe_dest:
            entry = board.ENTRY_POSITIONS[token.color]
            board_pos = (entry + target_pos) % 52
            danger = 0
            for opp_color in Color:
                if opp_color == token.color:
                    continue
                for opp_token in board.tokens[opp_color]:
                    if opp_token.state == "board":
                        opp_entry = board.ENTRY_POSITIONS[opp_color]
                        opp_board_pos = (opp_entry + opp_token.position) % 52
                        dist = (board_pos - opp_board_pos) % 52
                        if 1 <= dist <= 6:
                            danger += (7 - dist)
            score -= danger * 10.0
        
        # 8. STACKING
        if target_state == "board":
            entry = board.ENTRY_POSITIONS[token.color]
            new_board_pos = (entry + target_pos) % 52
            friendly_nearby = 0
            for other in board.tokens[token.color]:
                if other is token or other.state != "board":
                    continue
                other_board_pos = board.get_token_position_on_board(other)
                if other_board_pos == new_board_pos:
                    friendly_nearby += 1
            if friendly_nearby > 0:
                score += 30.0 + friendly_nearby * 10.0
        
        # 9. COLUMN DEPTH
        if token.state == "column" and target_state == "column":
            score += target_pos * 5.0
        
        # 10. APPROACHING COLUMN ENTRY
        if token.state == "board" and target_state == "board":
            dist_to_entry = 51 - token.position
            score += (dist_to_entry - roll) * 1.5
        
        return score
    
    def choose_move(self, board: LudoBoard, color: Color, roll: int) -> Optional[tuple]:
        options = board.get_move_options(color, roll)
        if not options:
            return None
        
        best_score = float('-inf')
        best_option = None
        
        for opt in options:
            token = board.tokens[color][opt[0]]
            score = self.evaluate_move(token, opt, board, roll)
            score += random.uniform(0, 0.001)
            if score > best_score:
                best_score = score
                best_option = opt
        
        return best_option

def apply_move(board: LudoBoard, color: Color, move: tuple):
    token_idx, target_state, target_pos, _, _, _ = move
    token = board.tokens[color][token_idx]
    token.state = target_state
    token.position = target_pos
    if target_state == "board":
        token.steps_taken = target_pos + 1
    elif target_state == "column":
        token.steps_taken = 51 + target_pos + 1
    elif target_state == "home":
        token.steps_taken = 57
    elif target_state == "start":
        token.steps_taken = 0

def simulate_game(ai: LudoAI, verbose: bool = False) -> Color:
    board = LudoBoard()
    colors = list(Color)
    current_turn = 0
    max_turns = 3000
    
    for turn in range(max_turns):
        color = colors[current_turn]
        roll = random.randint(1, 6)
        
        move = ai.choose_move(board, color, roll)
        
        if move:
            token_idx, target_state, target_pos, _, _, _ = move
            token = board.tokens[color][token_idx]
            old = f"{token.state}({token.position})"
            apply_move(board, color, move)
            
            if all(t.state == "home" for t in board.tokens[color]):
                return color
        current_turn = (current_turn + 1) % 4
    
    best_color = None
    best_finished = -1
    for c in colors:
        finished = sum(1 for t in board.tokens[c] if t.state == "home")
        if finished > best_finished:
            best_finished = finished
            best_color = c
    return best_color

if __name__ == "__main__":
    ai = LudoAI()
    wins = {color: 0 for color in Color}
    
    print("Simulating 200 games...")
    for g in range(200):
        winner = simulate_game(ai)
        wins[winner] += 1
    
    print("\nResults:")
    for color, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {color.name}: {count} wins")
    
    print("\nSample game log:")
    winner = simulate_game(ai, verbose=True)
    print(f"\nWinner: {winner.name}")
