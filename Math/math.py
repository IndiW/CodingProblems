'''
Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

    Starting with any positive integer, replace the number by the sum of the squares of its digits.
    Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
    Those numbers for which this process ends in 1 are happy.

Return true if n is a happy number, and false if not.

'''


class Solution:
    def isHappy(self, n: int) -> bool:
        # can also use a floyds cycle detection alg
        # with a slow pointer and fast pointer
        # when slow == fast, we either hit 1 or not 1 == cycle
        seen = set()
        n = str(n)
        while True:
            if n in seen:
                return False
            seen.add(n)
            s = 0
            for c in n:
                s += int(c)**2
            if s == 1:
                return True
            n = str(s)