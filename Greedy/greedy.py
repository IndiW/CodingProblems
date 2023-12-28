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
