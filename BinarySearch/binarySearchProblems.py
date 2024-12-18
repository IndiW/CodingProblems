'''
TEMPLATE


def bin_search(arr):\
    def condition(value):
        pass

    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left)//2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
'''

'''
278. First Bad Version

You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.
'''

# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        
        left, right = 0, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left


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


'''
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

    [4,5,6,7,0,1,2] if it was rotated 4 times.
    [0,1,2,4,5,6,7] if it was rotated 7 times.

Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

'''


class Solution:
    def findMin(self, nums: List[int]) -> int:
        lo = 0
        hi = len(nums) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            # if nums[mid] > nums[hi], the pivot point must be on the right of mid. 
            # otherwise it'll be on the left of mid (including mid)
            # eg [3,4,5,1,2] the 5 > 2, so pivot point is on right of 5
            # if nums[mid] <= nums[hi], its a sorted array and we can move hi back. 
            if nums[mid] > nums[hi]:
                lo = mid + 1
            else:
                hi = mid
        
        return nums[lo]


'''
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

'''

class Solution:
    def search(self, nums: 'List[int]', target: int) -> int:
        if len(nums) == 1:
            return 0 if nums[0] == target else -1
        
        lo = 0
        hi = len(nums) - 1

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] == target:
                return mid
            # for a number nums[mid]
            # if there is a number larger than nums[mid], it would be to the right of it
            # [1,2,3,4,5] [2,3,4,5,1] [3,4,5,1,2], [4,5,1,2,3], 
            # [5,1,2,3,4]
            # [4,5,6,7,0,1,2]
            if nums[mid] >= nums[lo]:
                # pivot is right of mid
                if target >= nums[lo] and target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
                
            else:
                # pivot is left of mid
                if target <= nums[hi] and target > nums[mid]:
                    lo = mid + 1
                else:
                    hi = mid - 1


        
        return -1



'''
Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

    TimeMap() Initializes the object of the data structure.
    void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
    String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".


'''
            

class TimeMap:

    def __init__(self):
        self.store = {}
        self.timestampStore = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key in self.store:
            self.store[key][timestamp] = value
            self.timestampStore[key].append(timestamp)
        else:
            self.store[key] = { timestamp: value }
            self.timestampStore[key] = [timestamp]

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        else:
            nums = self.timestampStore[key]
            target = timestamp
            lo = 0
            hi = len(nums) - 1
            prevTime = -1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if nums[mid] > target:
                    hi = mid - 1
                elif nums[mid] < target:
                    prevTime = nums[mid]
                    lo = mid + 1
                else:
                    return self.store[key][nums[mid]]
            if prevTime == -1:
                return ""
            
            return self.store[key][prevTime] 

        


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)



'''
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).


'''

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        is_even = (len(nums1) + len(nums2)) % 2 == 0
        # left partition is half of total elements ie m + n / 2
        # if we get left partition of the smaller array,
        # (m+n/2) - left_partition_smaller = right_partition_larger
        # check if left partition max of B is smaller than right partition min of A
        # if partition is incorrect, we update our binary search 
        # get the min and max of the middle elements
        A, B = nums1, nums2
        total = len(A) + len(B)
        half = total // 2
        if len(B) < len(A):
            A, B = B, A
        
        l, r = 0, len(A) - 1
        while True:
            i = (l + r) // 2 # A
            j = half - i - 2 # B, -2 because indexed at 0

            Aleft = A[i] if i >= 0 else float('-inf')
            Aright = A[i + 1] if (i+1) < len(A) else float('inf')

            Bleft = B[j] if j >= 0 else float('-inf')
            Bright = B[j+1] if (j+1) < len(B) else float('inf')

            if Aleft <= Bright and Bleft <= Aright:
                # partition is correct
                if total % 2:
                    return min(Aright, Bright)
                else:
                    return (min(Aright, Bright) + max(Aleft, Bleft)) / 2
            
            elif Aleft > Bright:
                r = i - 1
            else:
                l = i + 1
        
        


        

'''
1011. Capacity To Ship Packages Within D Days
A conveyor belt has packages that must be shipped from one port to another within days days.

The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.
'''
        
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        # must retain order of weights


        def feasible(capacity):
            days_count = 1
            total = 0
            for w in weights:
                total += w
                if total > capacity: # reached capacity, move to next day
                    total = w
                    days_count += 1
                    if days_count > days: # ran out of days
                        return False
            return True


        left, right = max(weights), sum(weights)
        # at minimum, a day needs to handle the heaviest weight ie max(weights) => left pointer
        # at most, a day would have to handle all weights ie sum(weights) => right pointer

        while left < right:
            mid = left + (right - left) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        
        return left





'''
69. Sqrt(x)


Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well.

You must not use any built-in exponent function or operator.

    For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.


'''

class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        if x <= 2:
            return 1
        left, right = 0, x
        while left < right:
            mid = left + (right - left) // 2
            if mid*mid > x:
                right = mid 
            else:
                left = mid + 1
        
        return left - 1


'''
35. Search Insert Position


Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.
'''

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        
        left, right = 0, len(nums)

        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1
        
        return left




'''
410. Split Array Largest Sum


Given an integer array nums and an integer k, split nums into k non-empty subarrays such that the largest sum of any subarray is minimized.

Return the minimized largest sum of the split.

A subarray is a contiguous part of the array.
'''

class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:

        def condition(mid):
            splits = 1
            total = 0
            for num in nums:
                total += num
                if total > mid:
                    splits += 1
                    total = num
                    if splits > k:
                        return False
            return True
        
        left, right = max(nums), sum(nums)
        while left < right:
            mid = left + (right - left) // 2
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        
        return left


'''
1482. Minimum Number of Days to Make m Bouquets

You are given an integer array bloomDay, an integer m and an integer k.

You want to make m bouquets. To make a bouquet, you need to use k adjacent flowers from the garden.

The garden consists of n flowers, the ith flower will bloom in the bloomDay[i] and then can be used in exactly one bouquet.

Return the minimum number of days you need to wait to be able to make m bouquets from the garden. If it is impossible to make m bouquets return -1.
'''

class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        if len(bloomDay) < m*k:
            return -1
        
        def condition(days_waited):
            bouquets = 0
            total_flowers = 0
            for days_required in bloomDay:
                if days_waited >= days_required:
                    total_flowers += 1
                    if total_flowers == k:
                        bouquets += 1
                        total_flowers = 0
                else:
                    total_flowers = 0
            return bouquets >= m



        left, right = 1, max(bloomDay)

        while left < right:
            mid = left + (right - left) // 2
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        
        return left


'''
NOTES

Amazing resource: https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/solutions/769698/python-clear-explanation-powerful-ultimate-binary-search-template-solved-many-problems

'''