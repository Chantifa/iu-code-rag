from typing import List
from functools import lru_cache


class Solution:
    def canCross(self, stones: List[int]) -> bool:
        # Edge case: if second stone is not at position 1, frog can't make first jump
        if len(stones) < 2 or stones[1] != 1:
            return False

        # Create a set for O(1) lookup of stone positions
        stone_set = set(stones)
        target = stones[-1]

        # Memoization using lru_cache
        @lru_cache(maxsize=None)
        def canReach(position: int, last_jump: int) -> bool:
            # If we reached the last stone, success!
            if position == target:
                return True

            # Try jumps of k-1, k, k+1
            for next_jump in [last_jump - 1, last_jump, last_jump + 1]:
                # Jump must be positive
                if next_jump > 0:
                    next_position = position + next_jump
                    # Check if there's a stone at the next position
                    if next_position in stone_set:
                        if canReach(next_position, next_jump):
                            return True

            return False

        # Start from position 1 (after first jump of 1 unit)
        return canReach(1, 1)


# Test cases
def test_solution():
    sol = Solution()

    # Example 1
    stones1 = [0, 1, 3, 5, 6, 8, 12, 17]
    result1 = sol.canCross(stones1)
    print(f"Example 1: stones = {stones1}")
    print(f"Output: {result1}")
    print(f"Expected: True")
    print(f"{'✓ PASS' if result1 == True else '✗ FAIL'}\n")

    # Example 2
    stones2 = [0, 1, 2, 3, 4, 8, 9, 11]
    result2 = sol.canCross(stones2)
    print(f"Example 2: stones = {stones2}")
    print(f"Output: {result2}")
    print(f"Expected: False")
    print(f"{'✓ PASS' if result2 == False else '✗ FAIL'}\n")

    # Additional test cases
    # Edge case: only two stones
    stones3 = [0, 1]
    result3 = sol.canCross(stones3)
    print(f"Test 3: stones = {stones3}")
    print(f"Output: {result3}")
    print(f"Expected: True")
    print(f"{'✓ PASS' if result3 == True else '✗ FAIL'}\n")

    # Edge case: gap too large at start
    stones4 = [0, 2]
    result4 = sol.canCross(stones4)
    print(f"Test 4: stones = {stones4}")
    print(f"Output: {result4}")
    print(f"Expected: False (first jump must be 1 unit)")
    print(f"{'✓ PASS' if result4 == False else '✗ FAIL'}\n")


if __name__ == "__main__":
    test_solution()