'''
Arrays
'''

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            if target - num in seen:
                return [seen[target - num], i]
            else:
                seen[num] = i
        
        return [-1, -1]
            

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        buy = prices[0]
        for price in prices:
            profit = max(profit, price - buy)
            if price < buy:
                buy = price
        
        return profit



# Moores voting algorithm for finding majority element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        candidate = 0

        for num in nums:
            if count == 0:
                candidate = num
            if num == candidate:
                count += 1
            else:
                count -= 1
        
        return candidate

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()

        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        # count number of zeros
        zeros = 0
            
        # move non zeros to front
        i = 0
        for num in nums:
            if num == 0:
                zeros += 1
            if num != 0:
                nums[i] = num
                i += 1
        
        # add zeros to end
        for zero in range(zeros):
            nums[i] = 0
            i += 1

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        ret = []

        while left <= right:
            if abs(nums[right]) > abs(nums[left]):
                ret.append(nums[right]**2)
                right -= 1
            else:
                ret.append(nums[left]**2)
                left += 1
        
        return ret[::-1]


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ret = set()
        nums.sort()

        for i in range(len(nums)):
            l = i+1
            r = len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    ret.add((nums[i],nums[l],nums[r]))
                    l += 1
                    r -= 1
                elif s > 0:
                    r -= 1
                else:
                    l += 1

        
        return list(ret)


        
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # [a,b,c,d]
    
        # [1,a,ab,abc] # prefix sum left to right
        # [1,d,cd,bcd] # prefix sum right to left

        # [abc,abd,acd,bcd]

        # [bcd,acd,abd,abc]

        ans = [1] * len(nums)

        left = 1
        for i in range(len(nums)):
            ans[i] *= left
            left *= nums[i]
        
        right = 1
        for i in range(len(nums)-1,-1,-1):
            ans[i] *= right
            right *= nums[i]
        
        return ans

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ret = []
        self.dfs(candidates, target, [], ret)
        return ret

    def dfs(self, nums, target, curr_combo, ret):
        if target < 0:
            return
        if target == 0:
            ret.append(curr_combo)
            return
        for i in range(len(nums)):
            self.dfs(nums[i:], target-nums[i], curr_combo+[nums[i]], ret)
                


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:        
        intervals.sort()

        ret = []
        for i in range(len(intervals)):
            if not ret or ret[-1][1] < intervals[i][0]:
                ret.append(intervals[i])
            else:
                ret[-1][1] = max(ret[-1][1], intervals[i][1])
        
        return ret


        
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # three pointers for 0s, 1s and 2s
        zeros = 0
        ones = 0
        twos = len(nums) - 1

        while ones <= twos:
            if nums[ones] == 0:
                # swap with last location of zero pointer
                nums[zeros], nums[ones] = nums[ones], nums[zeros]
                zeros += 1
                ones += 1
            elif nums[ones] == 1:
                # correct spot, increment
                ones += 1
            else:
                # found a two, swap with whatever is at 2s spot
                nums[ones], nums[twos] = nums[twos], nums[ones]
                twos -= 1
        

class Solution:
    def maxArea(self, height: List[int]) -> int:
        # two pointers
        # each iteration, compute max area
        # move pointer with lower height

        l, r = 0, len(height) - 1

        area = 0
        while l < r:
            area = max(area, min(height[l], height[r])*(r-l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        
        return area