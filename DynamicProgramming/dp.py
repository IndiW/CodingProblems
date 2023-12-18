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