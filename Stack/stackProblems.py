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

'''
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

    MinStack() initializes the stack object.
    void push(int val) pushes the element val onto the stack.
    void pop() removes the element on the top of the stack.
    int top() gets the top element of the stack.
    int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.

'''


class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if len(self.stack) == 0:
            self.stack.append([val, val])
        else:
            lastMin = self.stack[-1][1]
            if lastMin > val:
                self.stack.append([val, val])
            else:
                self.stack.append([val, lastMin])

    def pop(self) -> None:
        self.stack.pop(-1)
        

    def top(self) -> int:
        return self.stack[-1][0]
        

    def getMin(self) -> int:
        return self.stack[-1][1]