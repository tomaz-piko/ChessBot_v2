from abc import ABC, abstractmethod
from typing import Optional


class TimeControl(ABC):
    """
    Base class for chess time control strategies.
    
    All time control implementations should inherit from this class and
    implement the required methods.
    """
    
    @abstractmethod
    def get_move_time(self, is_ponder_hit: bool = False) -> float:
        """
        Calculate the amount of time to spend on the current move.
        
        Args:
            is_ponder_hit: Whether this move follows a successful ponder
            
        Returns:
            Time in seconds to spend on the current move
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the time control."""
        return self.__class__.__name__


class FixedTimeControl(TimeControl):
    """
    A simple time control strategy with fixed time for normal moves and ponder hits.
    """
    
    def __init__(self, normal_move_time: float, ponder_hit_time: Optional[float] = None):
        """
        Initialize with specific time limits.
        
        Args:
            normal_move_time: Time in seconds to spend on normal moves
            ponder_hit_time: Time in seconds to spend after a ponder hit.
                             If None, uses the same time as normal moves.
        """
        self.normal_move_time = normal_move_time
        self.ponder_hit_time = ponder_hit_time if ponder_hit_time is not None else normal_move_time
    
    def get_move_time(self, is_ponder_hit: bool = False) -> float:
        """
        Return the appropriate time based on whether this is a ponder hit.
        
        Args:
            is_ponder_hit: Whether this move follows a successful ponder
            
        Returns:
            Time in seconds to spend on the current move
        """
        if is_ponder_hit:
            return self.ponder_hit_time
        return self.normal_move_time
    
    def __str__(self) -> str:
        """String representation of the fixed time control."""
        if self.normal_move_time == self.ponder_hit_time:
            return f"FixedTimeControl({self.normal_move_time}s)"
        return f"FixedTimeControl({self.normal_move_time}s, ponder: {self.ponder_hit_time}s)"


class UniversalTimeControl(TimeControl):
    """
    Time control strategy for standard UCI mode with dynamic time allocation usefull for all time modes (bullet, rapid etc.).
    
    This time control simply estimates the number of moves in a game and allocates time based on that.
    It also considers the increment after each move if there is one and accounts for a small overhead for each move.
    In case of a ponder hit, if the pondered time is shorter than the time it would normally take, it makes up for the difference.
    If it has already pondered for more than it would normally take for a move, it uses the information gathered during pondering and does not allocate any time for the move.
    """
    
    def __init__(self,
                 moves_estimate: int = 100,
                 move_overhead_ms: int = 100):
        """
        Initialize with weights for different factors.
        
        Args:
            moves_estimate: Estimated total number of moves in the game
            move_overhead_ms: Expected overhead in milliseconds when making a move
        """
        self.moves_estimate = max(1, moves_estimate)
        self.move_overhead_ms = max(0, move_overhead_ms)
    
    def get_move_time(self, 
                      remaining_time_ms: int = 0, 
                      increment_ms: int = 0,
                      has_pondered_ms: int = 0, 
                      move_num: int = 1) -> float:
        """
        Calculate time to spend on the current move based on game state.
        
        Args:
            is_ponder_hit: Whether this move follows a successful ponder
            remaining_time_ms: Remaining time in milliseconds
            increment_ms: Time increment in milliseconds
            move_num: Current move number in the game
            
        Returns:
            Time in seconds to spend on the current move
        """
        remaining_moves = self.moves_estimate - move_num
        remaining_total_time = remaining_time_ms + (increment_ms - self.move_overhead_ms) * remaining_moves
        time_for_move = (remaining_total_time / remaining_moves)

        # If we already pondered for more than the calculate time,
        if has_pondered_ms > time_for_move:
            # If pondered time is greater than calculated move time, use pondered time
            time_for_move = 0
        elif has_pondered_ms > 0:
            # If pondered time is less than calculated move time, use remaining time
            time_for_move = time_for_move - has_pondered_ms

        time_for_move = min(time_for_move, remaining_time_ms)
        return (time_for_move - self.move_overhead_ms) / 1000.0 if time_for_move > 0 else 0 # Convert to seconds
        
    
    def __str__(self) -> str:
        """String representation of the UCI time control."""
        return f"AdaptiveTimeControl(est. moves per game={self.moves_estimate}, overhead={self.move_overhead_ms}ms)"


