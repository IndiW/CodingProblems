'''
Given an integer array nums of unique elements, return all possible
subsets
(the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.
'''

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ret = [[]]
        for n in nums:
            for r in ret[::]:
                ret.append(r + [n])
        return ret


'''
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the
frequency
of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

'''

class Solution:
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        ret = []
        self.dfs(candidates, target, [], ret)
        return ret

    def dfs(self, nums, target, path, ret):
        if target < 0:
            return
        if target == 0:
            ret.append(path)
            return
        for i in range(len(nums)):
            # nums[i:] is to skip the numbber num[i] that has been used to avoid duplication
            self.dfs(nums[i:], target - nums[i], path+[nums[i]], ret) 



'''
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
'''
# take a value, then compute the perm of the remaining values and combine the two
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ret = []
        def perm(nums, curr):
            nonlocal ret
            if len(nums) == 0:
                ret.append(curr)
                return
            for i in range(len(nums)):
                perm(nums[:i] + nums[i+1:], curr+[nums[i]])
        
        perm(nums, [])
        return ret

            
'''
Given an integer array nums that may contain duplicates, return all possible
subsets
(the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.
'''

# if duplicate, we don't need to add to all subsets, just the last subset
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        cur = []
        nums.sort()
        l = 0
        for i in range(len(nums)):
            # avoid duplicates by checking same values in sorted list
            if i > 0 and nums[i] == nums[i-1]:
                cur = [item + [nums[i]] for item in cur]
            else:
                cur = [item + [nums[i]] for item in res]
            res += cur
        
        return res
        
            
'''
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.
'''   
            