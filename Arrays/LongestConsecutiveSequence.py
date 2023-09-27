'''
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
You must write an algorithm that runs in O(n) time.

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9

0 <= nums.length <= 10^5
-10^9 <= nums[i] <= 10^9

'''

def longestConsecutive(nums) -> int:
    longest = 0
    unique = set(nums)

    for n in nums:
        # check if we're at the start of the possible sequence
        if n - 1 not in unique:
            curr = n
            currentSequence = 1
            while curr + 1 in unique:
                curr += 1
                currentSequence += 1
            longest = max(longest, currentSequence)
    
    return longest


print(longestConsecutive([0,3,7,2,5,8,4,6,0,1]))