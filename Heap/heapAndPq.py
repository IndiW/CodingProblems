'''
You are given an array of integers stones where stones[i] is the weight of the ith stone.

We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash them together. Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:

    If x == y, both stones are destroyed, and
    If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.

At the end of the game, there is at most one stone left.

Return the weight of the last remaining stone. If there are no stones left, return 0.

 
'''

# Multiply by -1 to use min heap as maxheap
import heapq

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [stone*-1 for stone in stones]
        heapq.heapify(stones)
        while len(list(stones)) > 1:
            y,x = heapq.nsmallest(2, stones)
            heapq.heappop(stones)
            heapq.heappop(stones)
            if y != x:
                heapq.heappush(stones, y-x)
        if len(stones) > 0:
            return list(stones)[0]*-1
        return 0


'''
Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Implement KthLargest class:

    KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of integers nums.
    int add(int val) Appends the integer val to the stream and returns the element representing the kth largest element in the stream.


'''

import heapq
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.pool = nums
        self.k = k
        heapq.heapify(self.pool)
        while len(self.pool) > k:
            # remove elements until we only have k elements
            heapq.heappop(self.pool)
        

    def add(self, val: int) -> int:
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, val)
        elif val > self.pool[0]:
            heapq.heapreplace(self.pool, val)
        return self.pool[0]
        


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)