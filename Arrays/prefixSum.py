'''
Given an integer array nums and an integer k, return true if nums has a good subarray or false otherwise.

A good subarray is a subarray where:

    its length is at least two, and
    the sum of the elements of the subarray is a multiple of k.

Note that:

    A subarray is a contiguous part of the array.
    An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.


'''

class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        # brute force, O(n^3)
        # for i in range(len(nums)):
        #     for j in range(i+1, len(nums)-1):
        #         if k % sum(nums[i:j]) == 0:
        #             return True
        # return False

        # prefix sum is an array such that 
        # prefix[i] = arr[0] + arr[1] + arr[2] ... + arr[i]
        # or prefix[i] = prefix[i-1] + arr[i]
        # we create the prefix sum array by starting with the
        # first value in the array, then adding the previous prefix sum
        # to the next value in the array 

        # to get the sum of the subarray from i+1 to j inclusive, you can compute
        # prefix_j - prefix_i
        # prefix_i = arr[0] + arr[1] + ... + arr[i]
        # prefix_j = arr[0] + arr[1] + ... + arr[j]
        # prefix_j - prefix_i = arr[j] + arr[j-1] + ... + arr[i+1] since everything else cancels out

        # sum(nums[i:j]) = prefix[j] - prefix[i]
        # (prefix[j] - prefix[i]) % k = 0

        prefixSum = [0]
        for i, num in enumerate(nums):
            prefixSum.append(num + prefixSum[i])
        
        # prefix[j] % k = prefix[i] % k
        prefix_mod = []
        for p in prefixSum:
            prefix_mod.append(p%k) # store prefix[x]%k

        # use a hashmap for O(1) lookup and insertion time
        mod_seen = { }
        for i in range(len(nums)):
            if prefix_mod[i] in mod_seen:
                # subarray must be at least len 2
                if i - mod_seen[prefix_mod[i]] > 1:
                    return True
            else:
                mod_seen[prefix_mod[i]] = i
        
        return False
            

