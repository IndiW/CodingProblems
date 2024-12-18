'''
31. Next Permutation

A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

    For example, for arr = [1,2,3], the following are all the permutations of arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].

The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).

    For example, the next permutation of arr = [1,2,3] is [1,3,2].
    Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
    While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger rearrangement.

Given an array of integers nums, find the next permutation of nums.

The replacement must be in place and use only constant extra memory.
'''

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # if array is in decending order, return reversed array
        # to create next largest perm, replace a[i-1] with the number 
        # which is larger than itself that is to the right, ie a[j]
        # note nums can be a perm at any stage

        # [1,4,5,8]
        # [1,4,8,5]
        # [1,5,4,8]
        # [1,5,8,4]
        # [1,8,4,5]
        # [1,8,5,4]
        # [4,1,5,8]
        # ...

        # scan from right (decrement index)
        # find pair where a[i] > a[i-1]. All numbers to the right of a[i-1] are sorted in decending order
        # swap a[i-1] with the smallest number greater than it to the right
        # reverse the subarray from a[i:]

        i = len(nums) - 2 # 1 element before last

        # find index of element we need to swap. first element that isn't increasing (from right to left)
        while i >= 0 and nums[i+1] <= nums[i]:
            i -= 1
        
        if i < 0: # we exhausted list and was entirely increasing (from right to left) so just reverse 
            self.reverse(nums, i+1)
            return
        else:
            # find first element that is greater than swap element, looking from right to left
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            self.swap(nums, i, j)
            self.reverse(nums, i+1)
            return


    def reverse(self, nums, start):
        i, j = start, len(nums) - 1

        while i < j:
            self.swap(nums, i, j)
            i += 1
            j -= 1
    
    def swap(self, nums, i, j):
        tmp = nums[i]
        nums[i] = nums[j]
        nums[j] = tmp

        






