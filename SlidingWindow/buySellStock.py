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