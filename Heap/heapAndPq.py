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
    

'''
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).

'''
# faster algorithm: put in heap and pop k times. 
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # d = sqrt((x1-x2)**2+(y1-y2)**2)
        #   = (x1**2)+(y1**2)**0.5
        #   => (x1**2) + (y1**2)
        distances = []
        # O(n)
        for point in points:
            distance = (point[0]**2)+(point[1]**2)
            distances.append([distance,point[0],point[1]])

        # O(nlogn)
        distances.sort()
        ret = []
        # O(n)
        for i in range(k):
            ret.append([distances[i][1],distances[i][2]])
        return ret
        

'''
Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Can you solve it without sorting?
'''

import random
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if not nums: return
        pivot = random.choice(nums)
        right = [x for x in nums if x > pivot]
        mid = [x for x in nums if x == pivot]
        left = [x for x in nums if x < pivot]
        L, M, R = len(left), len(mid), len(right)

        # [...left..., pivot, ...right...]
        if k <= R:
            # k is in right array
            return self.findKthLargest(right, k)
        elif k > R + M:
            # k is in left array
            return self.findKthLargest(left, k - R - M)
        else:
            return mid[0]

'''
Given a characters array tasks, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer n that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least n units of time between any two same tasks.

Return the least number of units of times that the CPU will take to finish all the given tasks.

'''