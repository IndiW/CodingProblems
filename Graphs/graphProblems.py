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
        # check the borders. Any Os connected to the corner is not surrounded. We mark them as .
        # all remaining Os are surrounded.
        # in the end we have . and O, the . are converted to O and the Os are converted to X

        def dfs(i, j):
            # mark all Os within the boundaries as .
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



'''
You are given an m x n grid where each cell can have one of three values:

    0 representing an empty cell,
    1 representing a fresh orange, or
    2 representing a rotten orange.

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.

'''


class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        # 1. find all rotten oranges
        # 2. dfs out from each rotten orange. Increment time on each recursion
        # 3. check if a fresh orange exists
        if not grid:
            return -1
        q = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 2:
                    q.append((i+1,j,1))
                    q.append((i-1,j,1))
                    q.append((i,j+1,1))
                    q.append((i,j-1,1))
        
        minutes = 0
        while q:
            i,j,k = q.pop(0)
            if 0 <= i < len(grid) and 0 <=j < len(grid[0]):
                if grid[i][j] == 1:
                    grid[i][j] = 2
                    minutes = max(minutes, k)
                    q.append([i+1,j,k+1])
                    q.append([i-1,j,k+1])
                    q.append([i,j+1,k+1])
                    q.append([i,j-1,k+1])
        

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    return -1
        
        return minutes

'''
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return true if you can finish all courses. Otherwise, return false.

'''

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # create hash where keys are courses and values are array of 
        # connected nodes
        # use dfs to see if we can visit all nodes
        # if we find cycle, we return false
        # nodes not visited are marked as 0
        # nodes being visited are marked as -1. if we find a -1 we foud a loop
        # when a node is visited, we mark as 1. 
        graph = [[] for _ in range(numCourses)]
        visited = [0 for _ in range(numCourses)]

        # add nodes to graph
        for prereq in prerequisites:
            x, y = prereq
            graph[x].append(y)
    

        def dfs(i):
            if visited[i] == 1:
                # already visited, don't need to revisit
                return True
            if visited[i] == -1:
                # a cycle exists
                return False
            else:
                visited[i] = -1
                for node in graph[i]:
                    if not dfs(node):
                        return False
                visited[i] = 1
                return True
        
        for course in range(numCourses):
            if not dfs(course):
                return False
        
        return True

        
                
                    

'''
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

'''

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        order = []
        # construct graph
        graph = [[] for _ in range(numCourses)]
        seen = [0 for _ in range(numCourses)]

        for prereq in prerequisites:
            x, y = prereq
            graph[x].append(y)
        
        # each course x in graph has an array of courses you have to take before you can take x

        # for each course, check their prereqs
        def dfs(course):
            # if we are already doing the course, we have a loop
            if seen[course] == -1:
                return False
            
            # if we've already done the course, nothing more to do
            if seen[course] == 1:
                return True
            
            # set current course to ongoing
            seen[course] = -1

            prereqs = graph[course]

            for prereq in prereqs:
                # if we can't do a prereq, return False
                if not dfs(prereq):
                    return False
            # after doing all the prereqs, complete the course
            seen[course] = 1
            # at this stage we can do the course
            order.append(course)
            return True
        
        for course in range(numCourses):
            if not dfs(course):
                return []
        
        return order



'''
In this problem, a tree is an undirected graph that is connected and has no cycles.

You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph.

Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.
'''

# Need to find the cycle 
# return one edge in the cycle


# why does this work
# try: for each edge u,v use dfs to traverse the graph and check if we can reach v from u. If we can its a duplicate
class Solution(object):
    def findRedundantConnection(self, edges):
        graph = collections.defaultdict(set)

        def dfs(source, target):
            if source not in seen:
                seen.add(source)
                if source == target: return True
                return any(dfs(nei, target) for nei in graph[source])

        for u, v in edges:
            seen = set()
            if u in graph and v in graph and dfs(u, v):
                return u, v
            graph[u].add(v)
            graph[v].add(u)

# union find is more efficient, but more complicated implementation.
# union find is a data structure that keeps track of elements which are split into one or more disjoint sets. It has two main operations: find and union. Find returns the representative of the set that an element is in. Union merges two sets into one.
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        parent = [-1] * (len(edges) + 1)
        rank = [0] * (len(edges) + 1)

        def find(x): # clasic find algorithm
            if parent[x] == -1: # the value is its own parent
                return x
            parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return False
            elif rank[root_x] < rank[root_y]: # union by rank. We add the smaller tree to the larger one to keep the trees flat
                parent[root_x] = root_y # union by updating the parent of root_x
                rank[root_y] += 1
                return True
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
                return True

        for x, y in edges:
            if not union(x, y): 
                return [x, y]
        
        raise ValueError("Illegal input.")
    


'''
Two sets are called disjoint sets if they don’t have any element in common.
Union find algorithm = disjoint set data structure 


'''
'''
Add system design problems to this repo?

'''

'''
A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

    Every adjacent pair of words differs by a single letter.
    Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
    sk == endWord

Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
'''

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # bfs from beginWord to endWord
        # construct a graph - adj matrix out of wordList        
        def bfs(beginWord, endWord):
            visited = set()
            q = [(beginWord, 1)]
            while q:
                word, count = q.pop(0)
                if word not in visited:
                    visited.add(word)
                    if word == endWord:
                        return count
                    for i in range(len(word)):
                        s = word[:i] + "_" + word[i+1:]
                        neighbours = graph.get(s, [])
                        for w in neighbours:
                            if w not in visited:
                                q.append((w, count+1))
            return 0

            
                 
        def differsByOne(s1, s2):
            if len(s1) != len(s2):
                return False
            diffByOne = False
            for i in range(len(s1)):
                if s1[i] != s2[i]:
                    if diffByOne:
                        return False
                    diffByOne = True
            return diffByOne
        
        if beginWord == endWord:
            return 0
        def construct_dict(word_list):
            d = {}
            for word in word_list:
                for i in range(len(word)):
                    s = word[:i] + "_" + word[i+1:]
                    d[s] = d.get(s, []) + [word]
            return d
        
        graph = construct_dict(wordList)
        
        return bfs(beginWord, endWord)

'''
You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin with "JFK". If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

    For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].

You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.

'''

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # tickets is an array of edges in a graph
        # need to find a path that also has smallest lexical order

        # construct adjacency matrix
        # use dfs on matrix, start at each node
        # Hierholzer's Algorithm

        graph = {}
        for ticket in tickets:
            if ticket[0] in graph:
                graph[ticket[0]].append(ticket[1])
            else:
                graph[ticket[0]] = [ticket[1]]
        
        # Sort children list in descending order so that we can pop last element 
        # instead of pop out first element which is costly operation
        for src in graph.keys():
            graph[src].sort(reverse=True)
        
        # dfs
        stack = ["JFK"]
        ret = []
        while len(stack) > 0:
            airport = stack[-1]
            if airport in graph and len(graph[airport]) > 0:
                stack.append(graph[airport].pop())
            else:
                # we reach an airport where we can't fly out of
                # this must be the last airport
                ret.append(stack.pop())
        
        return ret[::-1]


            
'''
You are given an array points representing integer coordinates of some points on a 2D-plane, where points[i] = [xi, yi].

The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.

Return the minimum cost to make all points connected. All points are connected if there is exactly one simple path between any two points.
'''

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # minimum spanning tree 
        def distance(a,b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        ans, number_of_points = 0, len(points)
        seen = set() # keep track of seen vertices
        vertices = [(0,(0,0))] # (weight, edge)

        while len(seen) < number_of_points:
            w, (u,v) = heapq.heappop(vertices) # u and v are the indexes of the verticies.
            if v in seen: continue
            ans += w
            seen.add(v)
            for i in range(number_of_points): # computing the weight of the edges from each point. Taking the smallest one using heappop 
                if i not in seen and i != v: 
                    heapq.heappush(vertices, (distance(points[i], points[v]),(v,i)))
        
        return ans


'''
You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

We will send a signal from a given node k. Return the minimum time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.

'''

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # dijsktra
        '''
    Create a set sptSet (shortest path tree set) that keeps track of vertices included in the shortest path tree, i.e., whose minimum distance from the source is calculated and finalized. Initially, this set is empty. 
    Assign a distance value to all vertices in the input graph. Initialize all distance values as INFINITE. Assign the distance value as 0 for the source vertex so that it is picked first. 
    While sptSet doesn’t include all vertices 
        Pick a vertex u that is not there in sptSet and has a minimum distance value. 
        Include u to sptSet. 
        Then update the distance value of all adjacent vertices of u. 
            To update the distance values, iterate through all adjacent vertices. 
            For every adjacent vertex v, if the sum of the distance value of u (from source) and weight of edge u-v, is less than the distance value of v, then update the distance value of v. 

        '''
        graph = {}
        heap = [(0,k)]

        for u,v,w in times:
            if u not in graph:
                graph[u] = [(v,w)]
            else:
                graph[u].append((v,w))
        visited = set()
        while heap:
            time, node = heapq.heappop(heap)
            visited.add(node)
            if len(visited) == n:
                return time
            if node not in graph:
                continue
            for v,w in graph[node]:
                if v not in visited:
                    heapq.heappush(heap, (time+w,v))
        return -1

'''
You are given an n x n integer matrix grid where each value grid[i][j] represents the elevation at that point (i, j).

The rain starts to fall. At time t, the depth of the water everywhere is t. You can swim from a square to another 4-directionally adjacent square if and only if the elevation of both squares individually are at most t. You can swim infinite distances in zero time. Of course, you must stay within the boundaries of the grid during your swim.

Return the least time until you can reach the bottom right square (n - 1, n - 1) if you start at the top left square (0, 0).
'''

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        # if grid[i][j] = n and we start at elevation 0, we have to wait
        # for n until we can swim to (i,j)
        # swim until we can't anymore
        # get the minimum wait time and try that path
        # recurse until we reach n-1,n-1

# do stuff




