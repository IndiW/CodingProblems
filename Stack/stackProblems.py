'''
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    Every close bracket has a corresponding open bracket of the same type.

'''

class Solution:
    def isValid(self, s: str) -> bool:
        
        stack = []
        pairs = { '(': ')', '{': '}', '[': ']'}
        for p in s:
            if p in pairs:
                stack.append(pairs[p])
            else:
                if not stack:
                    return False
                closing = stack.pop(-1)
                if closing != p:
                    return False
        
        return len(stack) == 0