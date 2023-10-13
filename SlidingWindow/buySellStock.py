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