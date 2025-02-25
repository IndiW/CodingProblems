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
                

