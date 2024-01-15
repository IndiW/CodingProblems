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

'''
You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

Increment the large integer by one and return the resulting array of digits.
'''

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if not digits:
            return digits
        digits[-1] += 1
        i = -1
        while digits[i] >= 10:
            digits[i] = digits[i] % 10
            i -= 1
            if i < -len(digits):
                digits = [1] + digits
                return digits
            else:
                digits[i] += 1
        
        return digits



        

'''
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
'''

def solve(mat):
    rev_mat = mat[::-1]

# reverse all the arrays
# swap diagonals


# [1,2,3]
# [4,5,6]
# [7,8,9]

# becomes

# [3,2,1]
# [6,5,4]
# [9,8,7]

# [7,4,1]
# [8,5,2]
# [9,6,3]