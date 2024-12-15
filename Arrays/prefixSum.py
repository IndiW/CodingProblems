'''
Prefix sum explained.

When we need to compute the sum of a contiguous sub array
Prefix sum is a new array where we store the running sum up until the index you are at
so p[2] = nums[0] + nums[1] + nums[2]
aka p[2] = nums[2] + p[1]

Using this knowledge, we can compute the sum of a contiguous sub array from i to j by doing
p[j] - p[i]

Usually these questions have some condition we want to check. This allows us to create some equation

Usually we want to do fast lookup of past p values, then we create a hash map to store and retrieve those values.


'''

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
            

'''
560. Subarray Sum Equals K

Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.
'''
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # generate prefix sum array
        # prefix_sum = [0]
        # for i, num in enumerate(nums):
        #     prefix_sum.append(prefix_sum[i] + num)

        # [1,2,3], k=3
        # [0,1,3,6]

        # [1], [0]
        # [0,1]
        # { 0, 1 }


        # sum[i:j] = prefix_sum[j] - prefix_sum[i]
        # prefix_sum[j] - prefix_sum[i] == k
        # prefix_sum[j] == k + prefix_sum[i]
        # prefix_sum[i] = prefix_sum[j] - k, can track running prefixSum
        prefix_sum = 0
        seen = { 0: 1}
        count = 0
        for num in nums:
            prefix_sum += num # get next prefix sum
            if (prefix_sum - k in seen): # from equation, and we add all instances of value prefix_sum[i]
                count += seen[prefix_sum-k]
            if prefix_sum in seen: # another instance of prefix_sum[i] is seen
                seen[prefix_sum] += 1
            else: # add to map for easy lookup
                seen[prefix_sum] = 1


        return count
        

'''
974. Subarray Sums Divisible by K

Given an integer array nums and an integer k, return the number of non-empty subarrays that have a sum divisible by k.

A subarray is a contiguous part of an array.
'''

class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        count = 0
        seen = {0:1}
        prefix = 0

        for num in nums:
            prefix += num
            mod = prefix%k
            if mod < 0:
                mod += k
            if mod in seen:
                count += seen[mod]
            if mod in seen:
                seen[mod] += 1
            else:
                seen[mod] = 1
    
        return count


