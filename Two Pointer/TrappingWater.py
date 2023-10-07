'''
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

'''


# for leftMax and rightMax (the bar heights), the water level is determined by the smaller value (the shorter bar)
# if the right bar is greater, the water level is determined by the bars going from left to right
# if the left bar is greater, the water level is deteremined by bars going from right to left

def solution(height):
    # need at least 3 bars to fill water
    if len(height)<= 2:
        return 0
    
    total = 0
    
    #pointers for current bar where we can fill water
    i = 0
    j = len(height) - 1
    
    # bordering bars that will limit the water
    lmax = height[0]
    rmax = height[-1]
    
    while i <= j:
        # update the max bar heights if we find a higher bar
        if height[i] > lmax:
            lmax = height[i]
        if height[j] > rmax:
            rmax = height[j]
        
        # if the left border bar is less than the right border bar, lmax is the limiting bar. We can fill up to lmax - 
        # at this point we know, height[l] <= lmax and height[r] <= rmax
        # if lmax <= rmax, we compute the smallest height which at this moment would be height[l] since height[l] < lmax < rmax
        if lmax <= rmax:
            total += lmax - height[i]
            i += 1
            
        #fill water upto rmax level for index j and move j to the left
        else:
            total += rmax - height[j]
            j -= 1
            
    return total