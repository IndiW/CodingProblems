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
    unique = set(nums)
    longest = 0
    for n in nums:
        curr = n
        currLength = 1
        while (curr + 1) in unique:
            currLength += 1
            curr += 1
        longest = max(longest, currLength)


    return longest


print(longestConsecutive([0,3,7,2,5,8,4,6,0,1]))