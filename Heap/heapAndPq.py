'''
You are given an array of integers stones where stones[i] is the weight of the ith stone.

We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash them together. Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:

    If x == y, both stones are destroyed, and
    If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.

At the end of the game, there is at most one stone left.

Return the weight of the last remaining stone. If there are no stones left, return 0.

 
'''

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        sorted_stones = sorted(stones)
        while len(sorted_stones) > 1:
            y = sorted_stones[-1]
            x = sorted_stones[-2]
            if x == y:
                sorted_stones.pop(-1)
                sorted_stones.pop(-1)
            else:
                sorted_stones.pop(-2)
                sorted_stones[-1] = y - x
                sorted_stones = sorted(sorted_stones)
        
        if len(sorted_stones) < 1:
            return 0
        return sorted_stones[0]
