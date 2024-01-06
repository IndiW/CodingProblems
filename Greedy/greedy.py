'''
Given an integer array nums, find the
subarray
with the largest sum, and return its sum.
'''

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # if positive, we keep adding
        # if negative, we start over
        # dp[i] = dp[i-1] > 0 ? dp[i-1] : nums[i]
        if not nums:
            return float('-inf')
        if len(nums) == 1:
            return nums[0]

        prev = nums[0]
        max_val = prev
        for i in range(1, len(nums)):
            if prev > 0:
                prev = prev + nums[i]
            else:
                prev = nums[i]
            max_val = max(max_val, prev)
        return max_val


'''
You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

 
'''

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # work backwards
        # check if current index + jump can reach last position
        # greedily choose left most position
        last = len(nums) - 1
        for i in range(last-1,-1,-1):
            if i + nums[i] >= last:
                last = i
        
        return last <= 0


'''
You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].

Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:

    0 <= j <= nums[i] and
    i + j < n

Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].

'''
            
            
class Solution:
    def jump(self, nums: List[int]) -> int:
        # min jumps = biggest jumps
        # we can always reach n-1
        
        # for each jump, take max distance 
        l = r = 0
        jumps = 0
        while r < len(nums) - 1:
            jumps += 1
            furthest = max([i+nums[i] for i in range(l, r+1)])
            l, r = r+1, furthest
        
        return jumps
    
'''
There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique

'''