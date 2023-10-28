'''
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.
'''

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo = 0
        hi = len(nums) - 1

        while lo <= hi:
            mid = lo + ((hi - lo) // 2)
            if nums[mid] > target:
                hi = mid - 1
            elif nums[mid] < target:
                lo = mid + 1
            else:
                return mid
        
        return -1


'''
You are given an m x n integer matrix matrix with the following two properties:

    Each row is sorted in non-decreasing order.
    The first integer of each row is greater than the last integer of the previous row.

Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.

 


'''


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        def binSearch(l, target):
            lo = 0
            hi = len(l) - 1
            while lo <= hi:
                mid = lo + ((hi - lo) // 2)
                if l[mid] > target:
                    hi = mid - 1
                elif l[mid] < target:
                    lo = mid + 1
                else:
                    return mid
            return -1
        
        low_row = 0
        top_row = len(matrix) - 1
        while low_row <= top_row:
            mid_row = low_row + ((top_row - low_row)//2)
            if target >= matrix[mid_row][0] and target <= matrix[mid_row][-1]:
                return binSearch(matrix[mid_row], target) != -1
            elif target <= matrix[mid_row][0]:
                top_row = mid_row - 1
            else:
                low_row = mid_row + 1

        return False
        
        
'''
Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

'''        

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        '''
        k such that sum(time to all the piles) <= h
        
        have to try numbers from 1 to max(piles)
        can use binSearch to find the number
        '''
        # we can only eat at most 1 pile per hour
        if (len(piles)) > h:
            return -1
        
        def validPile(k, h):
            time_to_eat = 0
            for pile in piles:
                time_to_eat += math.ceil(pile / k)
            return time_to_eat <= h
        
        lo = 1
        hi = max(piles)
        while lo < hi:
            mid = lo + ((hi - lo) // 2)
            if validPile(mid, h):
                hi = mid
            else:
                lo = mid + 1
        return hi



        
            





