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

'''
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

'''
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return []
        if n == 1:
            return ["()"]
        else:

            ret = []
            def gen(opened, closed, curr):
                if opened < closed:
                    return
                elif opened == n and closed < n:
                    gen(opened, closed+1, curr+")") 
                elif opened == n and closed == n:
                    ret.append(curr)
                    return
                elif opened > closed:
                    gen(opened+1, closed, curr + "(")
                    gen(opened, closed+1, curr+")") 
                elif opened == closed:
                    gen(opened+1, closed, curr+"(")
            gen(0, 0, "")
            return ret


'''
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

'''

class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        # monotonic stack
        # a stack where elements are always sorted
        # monotonic decreasing = sorted in desending order
        stack = []
        # sort the indices of the days
        ans = [0] * len(T)
        for i, t in enumerate(T):
            while stack and T[stack[-1]] < t:
                top = stack.pop()
                ans[top] = i - top
            stack.append(i)
        # if current day is not warmer than top of the stack, 
        # we push it to top
        # if current day is warmer, 
        # the number of days is the difference between the
        # current index, and the index on the top of the stack
        # we keep poping offthe stack until we reach a larger element

        # when we reach a colder day, we push current element

        return ans


'''
There are n cars going to the same destination along a one-lane road. The destination is target miles away.

You are given two integer array position and speed, both of length n, where position[i] is the position of the ith car and speed[i] is the speed of the ith car (in miles per hour).

A car can never pass another car ahead of it, but it can catch up to it and drive bumper to bumper at the same speed. The faster car will slow down to match the slower car's speed. The distance between these two cars is ignored (i.e., they are assumed to have the same position).

A car fleet is some non-empty set of cars driving at the same position and same speed. Note that a single car is also a car fleet.

If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.

Return the number of car fleets that will arrive at the destination.

'''

class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        positions = sorted(zip(position, speed))
        stack = []
        # reverse order would go from closest to furthest
        for pos, vel in positions[::-1]:
            dist = target - pos
            time = dist / vel #time to travel remaining distance
            if not stack: # first car
                # add to fleet
                stack.append(time)
            elif time > stack[-1]: 
                '''
                if the time it takes for the next car
                to reach the target is greater than 
                the car in front, it won't join 
                the 'car in front's fleet
                '''
                stack.append(time)
            
            else:
                # if the time is less than the car in front, 
                # the car behind is faster
                # thus it gets blocked by car in front
                # thus it joins that fleet
                continue
        return len(stack)



            
            
        

        

        