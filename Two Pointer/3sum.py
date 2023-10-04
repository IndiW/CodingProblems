'''
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

'''

def solution(nums):
    nums = sorted(nums)
    ret = set()
    if len(nums) == 3 and sum(nums) == 0:
        return [nums]
    i = 0
    while i < (len(nums) - 3):
        j = i + 1
        k = len(nums) - 1
        while j < k:
            s = nums[i] + nums[j] + nums[k]
            if s < 0:
                j += 1
            elif s > 0:
                k -= 1
            else:
                ret.add((nums[i], nums[j], nums[k]))
                if nums[k-1] == nums[k]:
                    k -= 1
                elif nums[j+1] == nums[j+1]:
                    j += 1
                else:
                    j += 1
                    k -= 1
        i += 1
    return list(ret)