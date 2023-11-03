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