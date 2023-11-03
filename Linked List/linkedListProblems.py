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



        
        