'''
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.
'''

# find location for interval baased on linear search
# combine intervals if necessary by comparing start with ends


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        ret = []

        for interval in intervals:
            if interval[1] < newInterval[0]:
                ret.append(interval)
            elif interval[0] > newInterval[1]:
                ret.append(newInterval)
                newInterval = interval
            elif interval[1] >= newInterval[0] or interval[0] <= newInterval[1]:
                newInterval[0] = min(interval[0], newInterval[0])
                newInterval[1] = max(interval[1], newInterval[1])
        
        ret.append(newInterval)

        return ret

'''
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

'''

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        ret = []
        if not intervals:
            return []
        intervals.sort()
        newInterval = intervals[0]
        for interval in intervals:
            if interval[1] < newInterval[0]:
                ret.append(interval)
            elif interval[0] > newInterval[1]:
                ret.append(newInterval)
                newInterval = interval
            elif interval[1] >= newInterval[0] or interval[0] <= newInterval[1]:
                newInterval[0] = min(interval[0], newInterval[0])
                newInterval[1] = max(interval[1], newInterval[1])
        
        ret.append(newInterval)
        return ret
            
                    



