'''
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

'''

class Solution:
    def maxProfit(self, prices) -> int:
        profit = 0
        currMin = prices[0]
        for price in prices:
            currMin = min(price, currMin)
            profit = max(profit, price - currMin)
        
        return profit
            

'''
Longest substring without repeating characters
Given a string s, find the length of the longest
substring
without repeating characters.

'''

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = {}
        longest = 0
        left = 0
        right = 0
        while right < len(s):
            if s[right] in seen and seen[s[right]] == True:
                while seen[s[right]] == True:
                    seen[s[left]] = False
                    left += 1
            else:
                seen[s[right]] = True
                right += 1
            longest = max(longest, right - left)
        
        return longest



'''
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.


'''

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        seen = {}
        l = 0
        r = 0
        longest = 0
        currMax = 0
        while r < len(s):
            if s[r] not in seen:
                seen[s[r]] = 1
            else:
                seen[s[r]] += 1
            
            currMax = max(currMax, seen[s[r]])
            
            if (r - l + 1) - currMax > k:
                seen[s[l]] -= 1
                l += 1
            
            longest = max(longest, r - l + 1)
            r += 1
        
        return longest


'''
Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

In other words, return true if one of s1's permutations is the substring of s2.

'''
from collections import Counter

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        cntr, w = Counter(s1), len(s1)   

        for i in range(len(s2)):
            if s2[i] in cntr: 
                cntr[s2[i]] -= 1
            if i >= w and s2[i-w] in cntr: 
                cntr[s2[i-w]] += 1

            if all([cntr[i] == 0 for i in cntr]): # see optimized code below
                return True

        return False


'''
Given two strings s and t of lengths m and n respectively, return the minimum window
substring
of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

'''

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t or not s:
            return ""
        left = 0 # closes current window
        right = 0 # expands current window
        # expand until we have all desired characters
        # contract and save smallest window til now

        dict_t = Counter(t)
        required = len(dict_t) # number of unique characters in t

        currentUnique = 0 # track how many unique characters in t are present in current window
        window_counts = {} # keep count of unique characters in current window
        windowLength = float('inf')
        ansL = None
        ansR = None

        while right < len(s):
            c = s[right]
            window_counts[c] = window_counts.get(c, 0) + 1

            if c in dict_t and window_counts[c] == dict_t[c]:
                currentUnique += 1
            
            while left <= right and currentUnique == required:
                c = s[left]

                # save smallest window
                if right - left + 1 < windowLength:
                    windowLength = right - left + 1
                    ansL = left
                    ansR = right
                
                window_counts[c] -= 1
                if c in dict_t and window_counts[c] < dict_t[c]:
                    currentUnique -= 1
                
                left += 1
            right += 1
        return "" if windowLength == float("inf") else s[ansL:ansR+1]


'''
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.
'''

# Note we want the max value of each window
#  A monotonic queue is a data structure that supports efficient insertion, deletion, and retrieval of elements in a specific order, typically in increasing or decreasing order.

# whenever we encounter a new element x, we want to discard all elements that are less than x before adding x
# we need to store the indices instead of the elements themselves is that we need to detect when elements leave the window due to sliding too far to the right.

from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        res = []

        # add indexes of max values to dq
        # set first max value
        for i in range(k):
            # if we see a bigger value, we remove all values in dq
            while dq and nums[i] >= nums[dq[-1]]:
                dq.pop()
            dq.append(i)
        
        # at this point, dq has the index of the max value in the first window

        # add first max value to result
        res.append(nums[dq[0]])

        # iterate through remaining elements
        for i in range(k, len(nums)):
            # if the left element is outside the window, we remove it
            # since we start at k, the first element we ignore is 0
            if dq and dq[0] == i - k:
                dq.popleft()
            # again we find index of max value 
            while dq and nums[i] >= nums[dq[-1]]:
                dq.pop()

            dq.append(i)
            res.append(nums[dq[0]])

        return res

                




            
                