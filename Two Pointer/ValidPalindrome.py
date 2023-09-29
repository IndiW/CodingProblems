'''
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

 

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Example 3:

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.


'''

def solution(s):
    normalized = []
    for c in s:
        if c.isalnum():
            normalized.append(c.lower())   

    return normalized == normalized[::-1]   


def twoPointerSolution(s):
    start = 0
    end = len(s) - 1

    while start < end:
        while not s[start].isalnum() and start < end:
            start += 1
        while not s[end].isalnum() and start < end:
            end -= 1
        
        if s[start].lower() != s[end].lower():
            return False
        start += 1
        end -= 1
    
    return True

print(solution("A man, a plan, a canal: Panama") == True)