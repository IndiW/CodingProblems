'''
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.
'''

# find location for interval baased on linear search
# combine intervals if necessary by comparing start with ends


class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if not intervals:
            return [newInterval]
        
        # [1,3], [6,9]
        #   [2,4]
        #       [4,7]
        ret = []
        i = 0
        while i < len(intervals):
            interval = intervals[i]
            if interval[0] <= newInterval[0] <= interval[1]:
                intervals[i][1] = max(interval[1], newInterval[1])
                newInterval = intervals[i]
            elif interval[0] <= newInterval[1] <= interval[1]:
                intervals[i][0] = min(interval[0], newInterval[0])
                newInterval = intervals[i]
            elif newInterval[0] <= interval[0] and newInterval[1] >= interval[1]:
                i += 1
                continue
            elif interval[0] <= newInterval[0] and interval[1] >= newInterval[1]:
                i += 1
                continue
            elif interval[0] <= newInterval[0] and interval[1] <= newInterval[0]:
                ret.append(interval)
            elif newInterval[0] <= interval[0] and newInterval[1] <= interval[0]:
                ret.append(newInterval)
            i += 1
        
        if newInterval != intervals[-1]:
            ret.append(interval)
        
        return ret


# DOESNT WORK