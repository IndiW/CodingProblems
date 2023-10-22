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


'''
You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:

    The valid operators are '+', '-', '*', and '/'.
    Each operand may be an integer or another expression.
    The division between two integers always truncates toward zero.
    There will not be any division by zero.
    The input represents a valid arithmetic expression in a reverse polish notation.
    The answer and all the intermediate calculations can be represented in a 32-bit integer.

 

Example 1:

Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Example 2:

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6



'''


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        operations = {
            '+': lambda a, b: a + b, 
            '-': lambda a, b: b - a, 
            '/': lambda a, b: int(b/a), 
            '*': lambda a, b: a * b
            }
        for s in tokens:
            if s in operations:
                if len(stack) < 2:
                    return -1
                op1 = stack.pop(-1)
                op2 = stack.pop(-1)
                stack.append(operations[s](op1, op2))
            else:
                stack.append(int(s))
        
        return stack.pop()
