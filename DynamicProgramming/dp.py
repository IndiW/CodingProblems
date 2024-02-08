'''
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
'''

class Solution:
    def climbStairs(self, n: int) -> int:
        # base case: O(0) = 0, O(1) = 1, O(2) = 2
        # recursive case: O(n) = O(n-1) + 1 + O(n-2) + 2
        memo = {}
        d = [0,1,2]
        # O(n) = O(n-1) + O(n-2) + 1
        if n < len(d):
            return d[n]
        else:
            for i in range(3, n+1):
                d.append(d[i-1] + d[i-2])        
        
        return d[-1]

'''
You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index 0, or the step with index 1.

Return the minimum cost to reach the top of the floor.

'''

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # dp[i] = min(cost to get to i-1, cost to get to i-2) + cost[i]
        dp = []
        if not cost or cost == 1:
            return 0
        for i, c in enumerate(cost):
            if i <= 1:
                dp.append(c)
            else:
                val = min(dp[i-1], dp[i-2])+c
                dp.append(val)
        return min(dp[-1], dp[-2])


'''
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
'''

class Solution:
    def rob(self, nums: List[int]) -> int:
        # base case: nums = [] => 0
        # nums = [x] => x
        # nums = [x,y] => max(x,y)
        # nums = [x,y,z] => max(x+z,y)
        # recursive: dp[i] = max(dp[0] + dp[2:], dp[1:])
        # may have a nicer solution
        dp = []
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        prev = nums[0]
        prev2 = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            curr = max(prev + nums[i], prev2)
            dp.append(max(prev + nums[i], prev2))
        
        return dp[-1]

'''
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

'''

class Solution:
    def rob(self, nums: List[int]) -> int:
        def helper(nums):
            dp1, dp2 = 0,0
            for num in nums:
                dp1,dp2 = dp2, max(dp1 + num, dp2)
            return dp2
        
        return max(nums[0] + helper(nums[2:-1]), helper(nums[1:]))


'''
Given a string s, return the longest
palindromic
substring
in s.
'''

class Solution:
    def longestPalindrome(self, s: str) -> str:
        ret = ""
        
        def findPalin(l, r):
            nonlocal ret
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if r - l + 1 > len(ret):
                    ret = s[l:r+1]
                l -= 1
                r += 1

        for i in range(len(s)):
            l = r = i
            findPalin(l, r)

            l = i
            r = i + 1
            findPalin(l, r)
        
        return ret

            

'''

Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.

A substring is a contiguous sequence of characters within the string

'''

class Solution:
    def countSubstrings(self, s: str) -> int:
        def findPalin(i, j):
            count = 0
            while i >= 0 and j < len(s) and s[i] == s[j]:
                count += 1
                i -= 1
                j += 1
            return count
        
        ret = 0
        for i in range(len(s)):
            l = r = i
            ret += findPalin(l, r)

            l = i
            r = i + 1
            ret += findPalin(l, r)
        
        return ret
    


'''
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.
'''      

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # OPT(amount) = min(OPT(amount - coin) for coin in coins)
        dp = [0]
        for i in range(1, amount+1):
            counts = []
            for coin in coins:
                if len(dp) >= coin and dp[i-coin] >= 0:
                    counts.append(dp[i-coin] + 1)
                else:
                    counts.append(float('inf'))
            dp.append(min(counts))
        
        if dp[-1] == float('inf'):
            return -1
        return dp[-1]
        
