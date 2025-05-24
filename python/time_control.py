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


class AdaptiveTimeControl(TimeControl):
    """
    Time control strategy for standard UCI mode with dynamic time allocation.
    
    This time control uses information about remaining time, increments, and
    game state to determine how much time to spend on a move.
    """
    
    def __init__(self,
                 moves_estimate: int = 100,
                 move_overhead_ms: int = 100,
                 ponder_factor: float = 0.75):
        """
        Initialize with weights for different factors.
        
        Args:
            move_num_weight: Weight for the move number factor (0.0-1.0)
            piece_num_weight: Weight for the piece count factor (0.0-1.0)
            moves_estimate: Estimated total number of moves in the game
            move_overhead_ms: Expected overhead in milliseconds when making a move
            ponder_factor: Factor to reduce time on ponder hits (0.0-1.0)
        """
        self.moves_estimate = max(1, moves_estimate)
        self.move_overhead_ms = max(0, move_overhead_ms)
        self.ponder_factor = max(0.0, min(1.0, ponder_factor))
    
    def get_move_time(self, 
                      remaining_time_ms: int = 0, 
                      increment_ms: int = 0,
                      is_ponder_hit: bool = False, 
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

        if is_ponder_hit:
            time_for_move *= self.ponder_factor

        time_for_move = min(time_for_move, remaining_time_ms)
        return (time_for_move - self.move_overhead_ms) / 1000.0  # Convert to seconds
        
    
    def __str__(self) -> str:
        """String representation of the UCI time control."""
        return f"AdaptiveTimeControl(est. moves per game={self.moves_estimate}, overhead={self.move_overhead_ms}ms, ponder={self.ponder_factor})"


