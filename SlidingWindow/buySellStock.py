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