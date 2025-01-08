'''
Arrays
'''

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            if target - num in seen:
                return [seen[target - num], i]
            else:
                seen[num] = i
        
        return [-1, -1]
            

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        buy = prices[0]
        for price in prices:
            profit = max(profit, price - buy)
            if price < buy:
                buy = price
        
        return profit



# Moores voting algorithm for finding majority element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        candidate = 0

        for num in nums:
            if count == 0:
                candidate = num
            if num == candidate:
                count += 1
            else:
                count -= 1
        
        return candidate

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()

        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        # count number of zeros
        zeros = 0
            
        # move non zeros to front
        i = 0
        for num in nums:
            if num == 0:
                zeros += 1
            if num != 0:
                nums[i] = num
                i += 1
        
        # add zeros to end
        for zero in range(zeros):
            nums[i] = 0
            i += 1

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        ret = []

        while left <= right:
            if abs(nums[right]) > abs(nums[left]):
                ret.append(nums[right]**2)
                right -= 1
            else:
                ret.append(nums[left]**2)
                left += 1
        
        return ret[::-1]


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ret = set()
        nums.sort()

        for i in range(len(nums)):
            l = i+1
            r = len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    ret.add((nums[i],nums[l],nums[r]))
                    l += 1
                    r -= 1
                elif s > 0:
                    r -= 1
                else:
                    l += 1

        
        return list(ret)


        
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # [a,b,c,d]
    
        # [1,a,ab,abc] # prefix sum left to right
        # [1,d,cd,bcd] # prefix sum right to left

        # [abc,abd,acd,bcd]

        # [bcd,acd,abd,abc]

        ans = [1] * len(nums)

        left = 1
        for i in range(len(nums)):
            ans[i] *= left
            left *= nums[i]
        
        right = 1
        for i in range(len(nums)-1,-1,-1):
            ans[i] *= right
            right *= nums[i]
        
        return ans

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ret = []
        self.dfs(candidates, target, [], ret)
        return ret

    def dfs(self, nums, target, curr_combo, ret):
        if target < 0:
            return
        if target == 0:
            ret.append(curr_combo)
            return
        for i in range(len(nums)):
            self.dfs(nums[i:], target-nums[i], curr_combo+[nums[i]], ret)
                


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:        
        intervals.sort()

        ret = []
        for i in range(len(intervals)):
            if not ret or ret[-1][1] < intervals[i][0]:
                ret.append(intervals[i])
            else:
                ret[-1][1] = max(ret[-1][1], intervals[i][1])
        
        return ret


        
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # three pointers for 0s, 1s and 2s
        zeros = 0
        ones = 0
        twos = len(nums) - 1

        while ones <= twos:
            if nums[ones] == 0:
                # swap with last location of zero pointer
                nums[zeros], nums[ones] = nums[ones], nums[zeros]
                zeros += 1
                ones += 1
            elif nums[ones] == 1:
                # correct spot, increment
                ones += 1
            else:
                # found a two, swap with whatever is at 2s spot
                nums[ones], nums[twos] = nums[twos], nums[ones]
                twos -= 1
        

class Solution:
    def maxArea(self, height: List[int]) -> int:
        # two pointers
        # each iteration, compute max area
        # move pointer with lower height

        l, r = 0, len(height) - 1

        area = 0
        while l < r:
            area = max(area, min(height[l], height[r])*(r-l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        
        return area


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # possible starts i where gas[i] >= cost[i]
        def isValidRoute(i):
            if gas[i] < cost[i]:
                return False
            currentGas = gas[i] - cost[i]
            position = i + 1
            while position !== i:
                 
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        total_surplus = 0
        surplus = 0
        start = 0

        for i in range(n):
            total_surplus += gas[i] - cost[i]
            surplus += gas[i] - cost[i]


            # if we hit negative when we get to i, then we will
            # hit negative for every value from start to i
            # thus, set next possible start to i+1
            if surplus < 0:
                surplus = 0
                start = i + 1
            
        # surplus from k onwards needs to be more than the surplus from 0 to k
        # since we need to circle back to k

        # surplus[k:n] >= surplus[0:k]
        # total surplus = surplus[0:k] + surplus[k:n]
        # total_surplus - surplus[0:k] >= surplus[0:k]
        # total_surplus >= 0
        return -1 if total_surplus < 0 else start


class Solution:
    # should use union find. This is alternative solution
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)

        ret = 0

        for num in nums:
            # only check min values
            if num-1 not in nums:
                nex = num + 1
                while nex in nums:
                    nex += 1
                ret = max(ret, nex-num)
        return ret

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        '''
        # O(n) space, O(n) time
        k = k % len(nums)
        copy = nums[::]

        for i in range(len(nums)):
            if k + i >= len(nums):
                nums[i+k-len(nums)] = copy[i]
            else:
                nums[i+k] = copy[i]
        '''

        def reverse(nums, i, j):
            left = i
            right = j
            while left < right:
                temp = nums[left]
                nums[left] = nums[right]
                nums[right] = temp
                left += 1
                right -= 1
        
        k = k % len(nums)
        # partition is where first and last element meet
        # each rotation, we take last element and move to front
        # k rotations means we move last k elements to front
        # partition at last k elements ie len(nums) - 1 - k
        
        # reverse from 0 to partition
        reverse(nums, 0, len(nums) - 1 - k)
        # reverse from partition to end
        reverse(nums, len(nums) - k, len(nums) - 1)
        
        # reverse entire array
        reverse(nums, 0, len(nums) - 1)
        

class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        count = 0
        max_len = 0

        # the longest arr would be from 0 to the end
        # index 0 has equal zeros and ones
        table = {0:0}

        for i, num in enumerate(nums):
            if num == 0:
                count -= 1
            else:
                count += 1
            
            if count in table:
                max_len = max(max_len, i+1 - table[count])
            else:
                # e.g. [0,1] should be len 2
                # if we just use i, we'd be 1 since we're 0 indexed
                # thus increment by 1
                table[count] = i+1
        
        return max_len


            
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # prefix_j - prefix_i = sum(arr[i:j]) = k
        # k = prefix_j - prefix_i
        # prefix_j - k = prefix_i
        # [0,1,2,3]
    
        seen = {0:1}
        prefix = 0
        count = 0

        for i, num in enumerate(nums):
            prefix += num

            if (prefix - k) in seen:
                count += seen[prefix - k]
            
            if prefix in seen:
                seen[prefix] += 1
            else:
                seen[prefix] = 1
        
        return count

            


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        ret = sum(nums[:3])
        diff = abs(target - ret)

        for i in range(len(nums)-2):
            l = i + 1
            r = len(nums) - 1

            while l < r:
                summ = nums[i] + nums[l] + nums[r]
                new_diff = abs(target - summ)
                if new_diff < diff:
                    diff = new_diff
                    ret = summ
                if summ > target:
                    r -= 1
                elif summ == target:
                    return target
                else:
                    l += 1
        
        return ret
        
 

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # always take interval with earliest end time
        # if x is the earliest end time, 
        # rest of the intervals need to be greater than x
        # if we choose another interval y
        # and x <= y
        # there are less intervals than can be included with y
        # so going with earliest end time produces maximum capacity to 
        # hold remaining intervals
        # max capacity == minimum removals needed

        current_earliest_end = float('-inf')
        count = 0

        # sort by end time
        for start, end in sorted(intervals, key=lambda x: x[1]):
            if start >= current_earliest_end:
                # the interval is non overlapping
                current_earliest_end = end
            else:
                # the interval overlaps, so we remove it
                count += 1
        return count


import heapq

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # TODO
        # use heap
        # can get max in O(log(n))

        # can add next element to heap via push
        # need way of removing first element from heap
        # need way of getting max without popping


        ret = []
        if k >= len(nums):
            return [max(nums)]


        # convert all to negative to use minheap
        for i in range(len(nums)):
            nums[i] *= -1

        heap = nums[:k]

        heapq.heapify(heap)
        for i in range(k, len(nums)):
            min_val = heapq.heappop(heap)
            ret.append(min_val*-1)

            heapq.heappush(heap, nums[i])
    
        return ret



class Solution:
    def isValid(self, s: str) -> bool:
        while s:
            if '()' in s:
                s = s.replace('()', '')
            elif "{}" in s:
                s = s.replace('{}', '')
            elif '[]' in s:
                s = s.replace('[]', '')
            else:
                break
        
        return len(s) == 0

class Solution:
    def isValid(self, s: str) -> bool:
        map = { '(': ')', '{': '}', '[': ']' }

        stack = []

        for paren in s:
            if paren in map:
                stack.append(map[paren])
            else:
                if not stack:
                    return False
                closing = stack.pop(-1)
                if closing != paren:
                    return False
        
        return len(stack) == 0


class MyQueue:

    def __init__(self):
        self.s1 = []
        self.s2 = []
        

    def push(self, x: int) -> None:
        self.s1.append(x)
        

    def pop(self) -> int:
        self.peek()
        return self.s2.pop()
        

    def peek(self) -> int:
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]
        

    def empty(self) -> bool:
        return not self.s1 and not self.s2
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()


class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:

        def buildStack(s):
            stack = []
            for c in s:
                if c == '#':
                    if not stack:
                        continue
                    else:
                        stack.pop()
                else:
                    stack.append(c)
            return stack
        
        
        return buildStack(s) == buildStack(t)

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for token in tokens:
            if token in ['+', '-', '*', '/']:
                op1 = stack.pop()
                op2 = stack.pop()
                if token == '+':
                    stack.append(op1 + op2)
                elif token == '-':
                    stack.append(op2 - op1)
                elif token == '*':
                    stack.append(op2 * op1)
                elif token == '/':
                    stack.append(int(op2/op1))
            else:
                stack.append(int(token))
        
        return stack[-1]

class MinStack:

    def __init__(self):
        self.stack = []
        

    def push(self, val: int) -> None:
        if self.stack:
            top = self.stack[-1]
            self.stack.append([val, min(top[1], val)])
        else:
            self.stack.append([val, val])
        

    def pop(self) -> None:
        self.stack.pop()
        

    def top(self) -> int:
        return self.stack[-1][0]
        

    def getMin(self) -> int:
        return self.stack[-1][1]
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()



class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        if len(temperatures) == 1:
            return [0]
        stack = []
        ret = [0]*len(temperatures)

        for i, temp in enumerate(temperatures):
            while stack and stack[-1][1] < temp:
                top = stack.pop()
                ret[top[0]] = i-top[0]
            stack.append([i, temp])
    
        
        return ret


class Solution:
    def decodeString(self, s: str) -> str:
        # stack contains multiplier then previous string
        stack = []
        mult = 0
        curString = ''

        for c in s:
            if c == '[':
                # stop building mult
                # this mult will be used for next string
                stack.append((curString, mult))
                curString = ''
                mult = 0
            elif c == ']':
                # multiply our current string
                # add new string to old string = new current string
                (prevString, num) = stack.pop()
                curString = prevString + num*curString
            elif c.isdigit():
                # build mult as we see it
                mult = mult*10 + int(c)
            else:
                # build current string 
                curString += c
        return curString



class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:

        def willCollide(a, b):
            # collision only in this scenario
            return a > 0 and b < 0

        stack = []
        for ast in asteroids:
            while stack and willCollide(stack[-1], ast):
                # same size, so we lose the new asteroid and break
                if stack[-1] == -ast:
                    stack.pop()
                    break
                # bigger asteroid, so we remove last asteroid and keep going
                elif abs(ast) > abs(stack[-1]):
                    stack.pop()
                    continue
                # new asteroid gets destroyed. We break
                elif abs(ast) < abs(stack[-1]):
                    break
            else: # no collision
                stack.append(ast)
        return stack
                

class Solution:
    def calculate(self, s: str) -> int:
        if not s:
            return 0
        stack = [] # stores all sub computations
        num = 0 # current number we're working with
        sign = "+"

        for i in range(len(s)):
            if s[i].isdigit():
                # build next number
                num = num*10 + int(s[i])
            # when we see a new sign, we deal with the current number we're working with
            if s[i] in '+-*/' or i == len(s) - 1: # last value has to be added
                if sign == '-':
                    stack.append(-num)
                elif sign == '+':
                    stack.append(num)
                elif sign == '*':
                    stack.append(stack.pop()*num) # immediately multiply current number by last number seen because of BEDMAS
                else:
                    stack.append(int(stack.pop()/num)) # immediately divide last number seen by current number 
                sign = s[i]
                num = 0
        return sum(stack)

            

                
                
            
                




        

        


