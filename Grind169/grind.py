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

            
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = [] # store index of heights
        res = 0
        for i in range(len(height)):
            # if the last height in stack is less than the current height, water can be contained
            while stack and height[stack[-1]] < height[i]:
                bottom = stack[-1] # possible lowest height between current wall and last last wall seen
                stack.pop()
                if stack:
                    res += (min(height[i], height[stack[-1]]) - height[bottom])*(i-stack[-1]-1)
            stack.append(i) # add to stack if the height is decreasing.
        return res
                
                
            
                




        

        
### linked lists


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = curr = ListNode(0)

        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        curr.next = l1 or l2
        return dummy.next
            

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False


        slow = fast = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        
        return False
    

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
        
        return prev

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        return slow

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # find midpoint
        # reverse 2nd half of list
        # start from beginning and mid and compare values

        slow = head
        fast = head
        prev = None

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        prev = slow # midpoint
        nxt = slow.next
        prev.next = None
        while nxt:
            tmp = nxt.next
            nxt.next = prev
            prev = nxt
            nxt = tmp
        fast, slow = head, prev
        while slow:
            if fast.val != slow.val: return False
            fast = fast.next
            slow = slow.next
        return True


# wip
class Node:
    def __init__(self, key, val):
        self.key = key
        self.value = val
        self.next = None
        self.prev = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dict = {}
        self.head = Node(0, 0) # head is always a dummy node
        self.tail = Node(0,0) # tail is always a dummy node
        self.head.next = self.tail
        self.tail.prev = self.head
    
        
    def get(self, key: int) -> int:
        if key in self.dict:
            node = self.dict[key]
            self._remove(node)
            self._add(node) # adds to the end 
            return node.value
        return -1

        
    def put(self, key: int, value: int) -> None:
        if key in self.dict:
            self._remove(self.dict[key])
        node = Node(key, value)
        self._add(node)
        self.dict[key] = node
        if len(self.dict) > self.capacity:
            node = self.head.next
            self._remove(node)
            del self.dict[node.key]
        
    
    def _remove(self, node):
        # remove a node by changing where the prev node and next node point to
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p
    
    # adds node to end of LL
    def _add(self, node):
        p = self.tail.prev # the tail is a dummy node. We get 1 before that - the 'last' node
        p.next = node # point the last node to the new node
        self.tail.prev = node # the tail dummy points to the new node
        node.prev = p # the new node's previous points to the 'last' node
        node.next = self.tail # the new nodes next is the dummy tail


            
        

        
        
            


        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        slow = fast = head
        for i in range(n):
            fast = fast.next
        if fast == None:
            return head.next
        
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
        
        slow.next = slow.next.next

        return head




            
        

        
        
            


        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next = head
        curr = dummy

        while curr.next and curr.next.next:
            first = curr.next
            second = curr.next.next
            curr.next = second
            first.next = second.next
            second.next = first
            curr = curr.next.next
        return dummy.next



        

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        
        odd = oddstart = head
        even = evenstart = head.next

        while even and even.next:
            nextodd = even.next
            nexteven = even.next.next

            odd.next = nextodd
            even.next = nexteven

            odd = nextodd
            even = nexteven
        
        odd.next = evenstart

        return oddstart


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        if not l1 or not l2:
            return l1 or l2

        l1head = l1
        l2head = l2

        while l1head and l2head:
            l1head = l1head.next
            l2head = l2head.next
        
        if not l1head:
            longer = l2
            shorter = l1
        else:
            longer = l1
            shorter = l2
        
        prev = ListNode(0)
        prev.next = longer
        ret = longer

        carry = 0
        while shorter:
            s = shorter.val + longer.val + carry
            if s >= 10:
                carry = 1
                longer.val = s - 10
            else:
                carry = 0
                longer.val = s
            longer = longer.next
            shorter = shorter.next
            prev = prev.next
        
        while carry and longer:
            s = longer.val + carry
            if s >= 10:
                carry = 1
                longer.val = s - 10
            else:
                carry = 0
                longer.val = s
            longer = longer.next
            prev = prev.next
        
        if carry:
            prev.next = ListNode(1)
    
        return ret
        

        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # find middle of list
        if not head or not head.next:
            return head
        fast = head.next
        slow = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        mid = slow.next
        slow.next = None
        l, r = self.sortList(head), self.sortList(mid)
        return self.merge(l, r)

    
    def merge(self, l, r):
        if not l or not r:
            return l or r
        dummy = p = ListNode(0)
        while l and r:
            if l.val < r.val:
                p.next = l
                l = l.next
            else:
                p.next = r
                r = r.next
            p = p.next
        
        p.next = l or r
        return dummy.next




            
            

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return head

        # find midpoint
        slow = fast = head
        
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next        

        # curr is 2nd half of list
        # reverse mid

        prev = None
        curr = slow.next

        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        
        slow.next = None
        
        # merge lists. Crisscross applesauce with changing pointer
        head1 = head
        head2 = prev

        while head2:
            tmp = head1.next
            head1.next = head2
            head1 = head2
            head2 = tmp
        
        


        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        def reverseList(head):
            prev = None
            curr = head
            while curr:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp
            return prev
            

        if k == 1:
            return head

        nodes = [head]
        dummy = ListNode(0)
        dummy.next = head
        count = k
        remaining = None


        # find each group of k nodes

        while dummy and dummy.next:
            start = dummy.next
            for i in range(k):
                if not dummy:
                    break
                dummy = dummy.next
            if not dummy:
                remaining = start
                break
            nodes.append(start)
            tmp = dummy.next
            dummy.next = None
            dummy = ListNode(0)
            dummy.next = tmp

        new_starts = []
        for i, node in enumerate(nodes):
            new_starts.append(reverseList(node))
            
        for i in range(len(nodes)-1):
            nodes[i].next = new_starts[i+1]
        
        nodes[-1].next = remaining

        return new_starts[0]
            


    



            
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        longest = 0

        l = 0
        r = 0
        count = 0

        while r < len(nums):
            if nums[r] == 0:
                count += 1
                while count > k:
                    if nums[l] == 0:
                        count -= 1
                    l += 1
            longest = max(longest, r - l + 1)
            r += 1
        
        return longest



            
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l = 0
        r = len(s) - 1

        while l < r:
            while l < r and not s[r].isalnum():
                r -= 1
            while l < r and not s[l].isalnum():
                l += 1
            
            
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        
        return True





class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        seen = {}

        for c in s:
            seen[c] = seen.get(c, 0) + 1
        
        for c in t:
            if c not in seen:
                return False
            seen[c] -= 1
            if seen[c] < 0:
                return False
        
        return True
        


class Solution:
    def longestPalindrome(self, s: str) -> int:
        seen = set()

        for c in s:
            if c in seen:
                seen.remove(c)
            else:
                seen.add(c)
        
        if len(seen) == 0:
            return len(s)
        return len(s) - len(seen) + 1


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        ans=""

        sorted_s = sorted(strs) # lexographic sort
        first = sorted_s[0]
        last = sorted_s[-1]

        for i in range(min(len(first), len(last))):
            if first[i] != last[i]:
                return ans
            ans += first[i]
        
        return ans


        
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # sliding window
        seen = set()
        l = 0
        longest = 0
        for r in range(len(s)):
            while s[r] in seen:
                seen.remove(s[l])
                l += 1
            
            seen.add(s[r])
            longest = max(longest, r - l + 1)
        
        return longest

class Solution:
    def longestPalindrome(self, s: str) -> str:
        longest = 0
        start = 0

        for i in range(len(s)):
            # i is the center of new potential palin
            right = i

            # get all equal characters 
            while right < len(s) and s[i] == s[right]:
                right += 1
            
            left = i - 1

            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            
            if right - left - 1 > longest:
                longest = right - left - 1
                start = left + 1 # because above loop breaks when we find unequal or go out of bounds
            
        return s[start:start+longest]
            


# todo optimize
from collections import Counter

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        org = Counter(p)
        n = len(p)
        ret = []

        for i in range(len(s) - n + 1):
            sub = s[i:i+n]
            new = Counter(sub)
            if org == new:
                ret.append(i)
        
        return ret
            

            


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = {}

        def getAnagramKey(s):
            key = [0]*26

            for c in s:
                key[ord(c) - ord('a')] += 1
            
            return tuple(key)


        for s in strs:
            key = getAnagramKey(s)
            if key in dic:
                dic[key].append(s)
            else:
                dic[key] = [s]
        
        return list(dic.values())


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # sliding window
        # longest substring = substring with x same characters + k different characters
        seen = {}
        longest = 0

        l = 0
        for r in range(len(s)):
            seen[s[r]] = seen.get(s[r], 0) + 1
            total_seen = seen.values()

            while sum(total_seen) - max(total_seen) > k:
                seen[s[l]] -= 1
                l += 1
            
            longest = max(longest, r - l + 1)
        
        return longest
        
            

import functools
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # Convert each integer to a string
        num_strings = [str(num) for num in nums]

        def string_compare(a, b):
            ab = a + b
            ba = b + a

            if ab > ba:
                return 1
            elif ab < ba:
                return -1
            else:
                return 0

        # Sort strings based on concatenated values
        num_strings.sort(key=functools.cmp_to_key(string_compare), reverse=True)

        # Handle the case where the largest number is zero
        if num_strings[0] == "0":
            return "0"

        # Concatenate sorted strings to form the largest number
        return "".join(num_strings)

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # t is always shorter
        if len(s) < len(t):
            return ""
        
        # track counts in shorter string
        char_count = defaultdict(int)
        for ch in t:
            char_count[ch] += 1
        
        target_chars_remaining = len(t) # needed to know when we have a valid window
        min_window = (0, float("inf")) 
        start_index = 0

        for end_index, ch in enumerate(s):
            # positive values in char_count means ch is required
            if char_count[ch] > 0:
                target_chars_remaining -= 1
            char_count[ch] -= 1

            # valid window
            if target_chars_remaining == 0:
                while True: # move left side of window conditionally
                    char_at_start = s[start_index]
                    if char_count[char_at_start] == 0: # exit loop because we need this char
                        break
                    # otherwise we don't need the char and we can shrink window
                    char_count[char_at_start] += 1
                    start_index += 1
                
                # update min window
                if end_index - start_index < min_window[1] - min_window[0]:
                    min_window = (start_index, end_index)
                
                # move left side of window while we can
                char_count[s[start_index]] += 1
                target_chars_remaining += 1
                start_index += 1
        
        return "" if min_window[1] > len(s) else s[min_window[0]:min_window[1]+1]




# Wip

class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:

        # { [word]: [array of indexes where word exists] }
        seen = {}
        for i in range(len(words)):
            if words[i] in seen:
                seen[words[i]].append(i)
            else:
                seen[words[i]] = [i]

        # check if reverse of word is in seen
        # or if reversed substring of word is in seen
        ret = []

        for i, word in enumerate(words):
            rev_word = "".join(reversed(word))
            if rev_word in seen:
                for ind in seen[rev_word]:
                    if i == ind: continue
                    ret.append([i, ind])
                    # ret.append([ind, i]) # optimization
            for j in range(len(word)):
                rev_sub = rev_word[:j]
                if rev_sub in seen:
                    for ind in seen[rev_sub]:
                        if i == ind: continue
                        ret.append([ind, i])
    
        return ret


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = [root]
        while q:
            node = q.pop()
            if not node:
                continue
            left = node.left
            right = node.right
            node.left = right
            q.append(right)
            node.right = left
            q.append(left)
        
        return root
    

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.dfs(root) != -1
    
    def dfs(self, root):
        if not root:
            return 0
        
        left = self.dfs(root.left)
        if left == -1: return -1
        right = self.dfs(root.right)
        if right == -1: return -1

        if abs(left - right) > 1:
            return -1

        return max(left, right) + 1        


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.diameter = 0
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def maxLength(node):
            if not node:
                return 0
            
            left = maxLength(node.left)
            right = maxLength(node.right)
            self.diameter = max(self.diameter, left + right)
            return 1 + max(left, right)
        
        maxLength(root)
        return self.diameter

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return p == q

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        return self.isMirror(root.left, root.right)
    
    
    def isMirror(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        
        if left.val == right.val:
            outter = self.isMirror(left.left, right.right)
            inner = self.isMirror(left.right, right.left)
            return outter and inner
        else:
            return False



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root:
            return False
        
        if self.isSameTree(root, subRoot):
            return True
        
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def isSameTree(self, root, subRoot):
        if root and subRoot:
            return root.val == subRoot.val and self.isSameTree(root.left, subRoot.left) and self.isSameTree(root.right, subRoot.right)
        return root is subRoot

# can also construct a hash representing the subtree and looking for a matching hash ^


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        q = [(root, 0)]
        ret = [[]]

        while q:
            node, level = q.pop(0)
            if level >= len(ret):
                ret.append([])

            ret[-1].append(node.val)
            if node.left:
                q.append((node.left, level+1))
            if node.right:
                q.append((node.right, level+1))

        return ret 
                

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # since p and q are unique, we won't find the same one twice
        if root == p or root == q:
            return root
        
        left = None
        right = None

        if root.left:
            left = self.lowestCommonAncestor(root.left, p, q)
        if root.right:
            right = self.lowestCommonAncestor(root.right, p, q)
        
        # we found a node in each subtree, so the common ancestor is the root
        if left and right:
            return root
        
        else:
            # we found the nodes in one subtree, so return whichever subtree it was
            return left or right

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        q = [(root, 0)]
        levels = [[]]

        while q:
            node, level = q.pop(0)
            
            if node:
                if level+1 > len(levels):
                    levels.append([])
                levels[-1].append(node)
                q.append((node.left, level+1))
                q.append((node.right, level+1))
            
        ret = []

        for level in levels:
            ret.append(level[-1].val)
        
        return ret

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # preorder - root, left, right
        # inorder - left, root, right

        if inorder:
 
            # pop will return -1 if preorder is empty
            ind = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[ind])
            
            root.left = self.buildTree(preorder, inorder[:ind])
            root.right = self.buildTree(preorder, inorder[ind+1:])

            return root




# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        ret = []

        def path(node, s, currPath):            
            if node:
                newSum = s + node.val
                newPath = currPath+[node.val]
                if newSum == targetSum and not node.left and not node.right:
                    ret.append(newPath)
                    return
                path(node.left, newSum, newPath)
                path(node.right, newSum, newPath)
        
        path(root, 0, [])

        return ret
            


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo = 0
        hi = len(nums) - 1

        while lo <= hi:
            mid = lo + ((hi - lo) // 2)
            if nums[mid] > target:
                hi = mid - 1
            elif nums[mid] < target:
                lo = mid + 1
            else:
                return mid
        
        return -1

           
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        q = deque()
        # node, level, index
        q.append((root, 1, 1))
        currLevel = 1
        startIndex = 1
        ret = 1

        while q:
            node, level, ind = q.popleft()
            if not node:
                continue
            if level > currLevel:
                startIndex = ind
                currLevel = level
                # in a binary heap array, child indexes are 2*ind and 2*ind+1 indexes away from their parent
            ret = max(ret, ind - startIndex + 1)
            if node.left:
                q.append((node.left, level+1, 2*ind))
            if node.right:
                q.append((node.right, level+1, 2*ind + 1))
        
        return ret
        

        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.result = 0
        # { sum: frequency }
        cache = { 0: 1 }

        self.dfs(root, targetSum, 0, cache)
        return self.result

    
    def dfs(self, node, target, currPathSum, cache):
        if not node:
            return
        
        currPathSum += node.val
        oldPathSum = currPathSum - target

        self.result += cache.get(oldPathSum, 0)
        cache[currPathSum] = cache.get(currPathSum, 0) + 1

        self.dfs(node.left, target, currPathSum, cache)
        self.dfs(node.right, target, currPathSum, cache)

        # we finished with this path, so we remove the current path sum (ie disregard this node for future)
        cache[currPathSum] -= 1






                
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        q = deque()
        initialLevel = 0
        q.append([root, initialLevel])
        levels = [[]]

        while q:
            node, level = q.popleft()    
            if node:
                if level >= len(levels):
                    levels.append([])
                if level % 2 == 0:
                    levels[-1].append(node.val)
                else:
                    levels[-1].insert(0, node.val)
                q.append((node.left, level+1))
                q.append((node.right, level+1))
        
        return levels


# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        
        left = 0
        right = n

        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        if not image:
            return image
        startColor = image[sr][sc]
        
        def dfs(x, y):
            if x >= len(image) or x < 0 or y >= len(image[0]) or y < 0:
                return
            if image[x][y] == color:
                return
            
            if image[x][y] != startColor:
                return
            
            image[x][y] = color
            dfs(x+1, y)
            dfs(x-1, y)
            dfs(x,y+1)
            dfs(x, y-1)
        
        dfs(sr, sc)

        return image
            




from collections import deque
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        rows = len(mat)
        cols = len(mat[0])

        q = deque()
        seen = set()

        for row in range(rows):
            for col in range(cols):
                if mat[row][col] == 0:
                    seen.add((row,col))
                    q.append((row, col))
        
        while q:
            x, y = q.popleft()

            for offset in [(1,0), (-1,0), (0,1), (0,-1)]:
                newX = offset[0] + x
                newY = offset[1] + y

                if newX >= 0 and newX < rows and newY >= 0 and newY < cols and (newX, newY) not in seen:
                    seen.add((newX, newY))
                    q.append((newX, newY))
                    mat[newX][newY] = mat[x][y] + 1
        
        return mat

                


# DP problems
class Solution:
    def climbStairs(self, n: int) -> int:
        arr = [0,1,2]
        if n < 3:
            return arr[n]
        
        prev = 2
        prevprev = 1
        curr = prev + prevprev
        for i in range(3, n+1):
            curr = prev + prevprev
            prevprev = prev
            prev = curr
        
        return curr


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # kadanes algorithm
        # intuition: when building a sum, we either add the next number or ditch our current sum
        # we would only ditch our current sum if our current sum + the next number is smaller than 
        # just going with the next number.

        if not nums:
            return 0
        
        curr_sum = max_sum = nums[0]

        for i in range(1, len(nums)):
            curr_sum = max(nums[i], curr_sum + nums[i]) # tracks the largest total so far
            max_sum = max(curr_sum, max_sum) # tracks all best totals we've seen
        
        return max_sum


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        # dp[coins, amount] = min(dp(coins, amount-coin[0]), dp(coins, amount-coin[1]), ...)

        if amount < 0:
            return -1

        dp = [0]

        for am in range(1, amount+1):
            if am in coins:
                dp.append(1)
            else:
                min_coin = float('inf')
                for coin in coins:
                    if am - coin >= 0:
                        min_coin = min(dp[am-coin] + 1, min_coin)
                dp.append(min_coin)
        
        if dp[-1] == float('inf'):
            return -1
        
        return dp[-1]
                

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ret = []
        def backtrack(curr, seen):
            # if current solution is valid, save it
            if len(curr) == len(nums):
                ret.append(curr)
                return
            
            # for all choices
            for num in nums:
                # if valid choice
                if num not in seen:
                    # backtrack while including that choice
                    seen.add(num)
                    backtrack(curr + [num], seen)
                    seen.remove(num)
            
        
        backtrack([], set())

        return ret


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ret = [[]]

        for num in nums:
            for arr in ret[::-1]:
                ret.append(arr + [num])
        
        return ret


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        map = {
            '1': '',
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        if not digits:
            return []

        ret = []
        
        def backtrack(i, curr):
            # if valid state, add to solution
            if i >= len(digits):
                ret.append(curr)
                return
            
            s = map[digits[i]]
            
            # for all valid choices, include the choice (curr + c) and we continue to next digit (i+1)
            for c in s:
                backtrack(i+1, curr+c)
        
        backtrack(0, "")
        return ret

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        
        ret = []

        def genPerms(curr, open, closed):
            # state is valid
            if len(curr) == n*2 and open == 0 and closed == 0:
                ret.append(curr)
                return
            
            # go through all choices
            # either you add an open or add a closed
            if open > 0:
                genPerms(curr+'(', open-1, closed)
            if open < closed:
                genPerms(curr+')', open, closed-1)

            
            

        genPerms("", n, n)

        return ret


class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        
        # create dict where each they key is
        # the first letter of each word
        # the value is the word
        # we iterate through characters in s
        # at each character, we take the words
        # that start with that character
        # remove the starting letter 
        # and add them to their new key in the dict
        # based on the first letter

        d = defaultdict(list)
        count = 0

        for word in words:
            d[word[0]].append(word)
        
        for c in s:
            words_starting_with_c = d[c]
            d[c] = []
            for word in words_starting_with_c:
                if len(word) == 1:
                    count += 1
                else:
                    d[word[1]].append(word[1:])
        
        return count

                

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # the numbers can have vals from 1-50
        # find number of occurrences of each number
        # 
        original = nums.count(k)
        max_gain = 0
        for m in range(1, 51):
            if m == k:
                continue
            curr = max_curr = 0
            for num in nums:
                if num == m:
                    curr += 1
                elif num == k:
                     curr -= 1
                curr = max(curr, 0) # ensure its positive and worth taking
                # if negative, we 'ditch' this curr and restart it at 0
                # this is how we consider subarrays
                max_curr = max(max_curr, curr) # biggest sub array so far
            max_gain = max(max_gain, max_curr) # max we can gain if we update number m
        return original + max_gain


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # [a, b], b first, then a
        # create a graph { course: [prereqs] }
        # dfs on each graph node
        # two sets for visited and visiting
        # true if you can visit each node
        # false if you hit cycle (visit a node you are visiting)

        finished = set()
        taking = set()

        # { 1: [4], 2: [4], 3: [1, 2], 4: []}
        courses = {}
        for course, prereq in prerequisites:
            if prereq not in courses:
                courses[prereq] = []
            if course in courses:
                courses[course].append(prereq)
            else:
                courses[course] = [prereq]
        

        def dfs(prereqs: List[int]):
            if not prereqs:
                return True
            ret = True
            for prereq in prereqs:
                if prereq in taking:
                    return False
                if prereq in finished:
                    continue
                taking.add(prereq)
                ret = ret and dfs(courses[prereq])
                taking.remove(prereq)
                finished.add(prereq)
            return ret
    
        ret = True
        for course, prereqs in courses.items():
            if course not in finished:
                ret = ret and dfs(prereqs)
                finished.add(course)
        
        return ret



                
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # island = 1s surrounded by water
        # count number of islands
        # step through array
        # if we see land, we bfs to visit all connected land
        # visit by setting value to 2
        # count each time new land is found

        if not grid:
            return 0

        n = len(grid)
        m = len(grid[0])
        def markLand(x, y):
            if x < 0 or x >= n or y < 0 or y >= m:
                return 
            if grid[x][y] == '0':
                return 
            
            grid[x][y] = '0'
            for offset_x, offset_y in [(0,1), (1,0), (0,-1), (-1,0)]:
                markLand(x+offset_x, y+offset_y)
            return
            


        count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    markLand(i, j)
                    count += 1
        
        return count

