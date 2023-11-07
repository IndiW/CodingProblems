'''
Given the head of a singly linked list, reverse the list, and return the reversed list.

'''

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        while head:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
        return prev

'''
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.


'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        start = head0 = ListNode(0)
        head1 = list1
        head2 = list2
        
        while head1 and head2:
            if head1.val > head2.val:
                head0.next = head2
                head2 = head2.next
            else:
                head0.next = head1
                head1 = head1.next
            head0 = head0.next
        
        if head1:
            head0.next = head1
        elif head2:
            head0.next = head2
        return start.next

'''
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln

Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return
        
        # find middle of linked list
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # slow is at middle of list, start of 2nd list
        # reverse second list
        prev, curr = None, slow
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        
        # prev is start of new reversed list
    
        # merge two lists
        first, second = head, prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next

          
'''
Given the head of a linked list, remove the nth node from the end of the list and return its head.

'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        fast, slow = head, head
        for _ in range(n): 
            fast = fast.next
        if not fast: 
            # when n == len(list), the nth node from the end is actually the first node
            return head.next
        while fast.next: 
            # the gap between fast and slow is n. When fast reaches the end, slow is at 1 before the node we want to remove
            fast, slow = fast.next, slow.next
        slow.next = slow.next.next
        return head

        

'''
A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

    val: an integer representing Node.val
    random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.

Your code will only be given the head of the original linked list.

'''


"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        visited = {}
        def dfs(head):
            if not head:
                return None
            if head in visited:
                return visited[head]
            node = Node(head.val)
            visited[head] = node
            node.next = dfs(head.next)
            node.random = dfs(head.random)
            return node

        return dfs(head)





        
        