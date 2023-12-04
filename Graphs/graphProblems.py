'''
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
'''

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        def bfs(x, y):
            if x >= len(grid) or x < 0 or y >= len(grid[0]) or y < 0:
                return
            if grid[x][y] == 'seen':
                return
            if grid[x][y] == '0':
                return
            if grid[x][y] == '1':
                grid[x][y] = 'seen'
                bfs(x+1, y)
                bfs(x-1,y)
                bfs(x,y+1)
                bfs(x,y-1)
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    count += 1
                    bfs(i,j)
        
        return count

