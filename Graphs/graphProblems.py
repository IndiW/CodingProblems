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

'''
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}

 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.
'''


"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return node
        seen = { node.val: Node(node.val)}

        q = [node]
        while q:
            curr = q.pop(-1)
            curr_clone = seen[curr.val]
            for nbr in curr.neighbors:
                if nbr.val not in seen:
                    seen[nbr.val] = Node(nbr.val)
                    q.append(nbr)
                curr_clone.neighbors.append(seen[nbr.val])

        return seen[node.val]



'''
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.
'''

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        maxArea = 0
        if not grid:
            return 0

        def bfs(x, y):
            nonlocal maxArea
            if x >= len(grid) or x < 0 or y >= len(grid[0]) or y < 0:
                return 0
            if grid[x][y] == 0:
                return 0
            else:
                grid[x][y] = 0
                area = 1
                area += bfs(x+1,y)
                area += bfs(x-1,y)
                area += bfs(x,y+1)
                area += bfs(x,y-1)
                return area
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    maxArea = max(bfs(i,j), maxArea)
        
        return maxArea

'''
There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.
'''

# not the most optimal.
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []

        def dfs(x, y):
            seen = {}
            stack = [(x,y)]
            isPacific = False
            isAtlantic = False
            while stack:
              coord = stack.pop(-1)
              if coord in seen:
                continue
              if coord[0] == 0 or coord[1] == 0:
                isPacific = True
              if coord[0] == len(heights) - 1 or coord[1] == len(heights[0]) - 1:
                isAtlantic = True
              
              if isPacific and isAtlantic:
                return [isPacific, isAtlantic]

              i = coord[0]
              j = coord[1]
              curr = heights[i][j]
              if i+1 <= len(heights)-1 and curr >= heights[i+1][j]:
                stack.append((i+1,j))
              if i-1 >= 0 and curr >= heights[i-1][j]:
                stack.append((i-1,j))
              if j+1 <= len(heights[0])-1 and curr >= heights[i][j+1]:
                stack.append((i,j+1))
              if j-1 >= 0 and curr >= heights[i][j-1]:
                stack.append((i,j-1))
              seen[coord] = True
            return [isPacific, isAtlantic]




        ret = []

        for i in range(len(heights)):
            for j in range(len(heights[0])):
                res = dfs(i, j)
                if res[0] and res[1]:
                    ret.append([i, j])
        
        return ret


'''
Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

 
'''

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # check the borders. Any Os connected to the corner is not surrounded
        # all remaining Os are surrounded

        def dfs(i, j):
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 'O':
                board[i][j] = '.'
                dfs(i+1,j)
                dfs(i-1,j)
                dfs(i,j+1)
                dfs(i,j-1)
        if not board or not board[0]:
            return
        for i in [0, len(board)-1]:
            for j in range(len(board[0])):
                dfs(i, j)
        for i in range(len(board)):
            for j in [0, len(board[0])-1]:
                dfs(i,j)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == '.':
                    board[i][j] = 'O'