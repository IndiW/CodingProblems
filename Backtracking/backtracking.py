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

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        ret = []
        def dfs(can, targ, curr, start):
            if targ == 0:
                ret.append(curr)
                return
            for i in range(start, len(can)):
                c = can[i]
                # skip dupes
                if i > start and can[i] == can[i-1]:
                    continue

                if c > targ: #we can do this since the array is sorted
                    break
                dfs(can, targ-c, curr+[c], i+1)
        
        dfs(candidates, target, [], 0)
        return ret
            
'''
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

'''

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i,j, word, board):
            if not word:
                return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[0]:
                return False
            tmp = board[i][j] #important part, we mark the cell as visited but then we need to restore it after
            board[i][j] = 'XX'
            ret = []
            for x,y in [[1,0],[-1,0],[0,1],[0,-1]]:
                ret.append(dfs(i+x,j+y,word[1:], board))
            board[i][j] = tmp
            return any(ret)

        if not board or not word:
            return False

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i,j, word,board[::]):
                    return True
        return False


'''
Given a string s, partition s such that every
substring
of the partition is a
palindrome
. Return all possible palindrome partitioning of s.
'''

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        self.dfs(s,[],res)
        return res

    
    def dfs(self,s,path,res):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s)+1):
            if s[:i] == s[:i][::-1]:
                self.dfs(s[i:], path+[s[:i]], res)
            


        
    
    
'''
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

'''
    
            
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        conv = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno",
               "7": "pqrs", "8": "tuv", "9": "wxyz"}

        
        if digits == "":
            return []
        
        rets = ['']
        for digit in digits:
            temp = []
            for letter in conv[digit]:
                for p in rets:
                    temp.append(p+letter)
            rets = temp
        return rets

            