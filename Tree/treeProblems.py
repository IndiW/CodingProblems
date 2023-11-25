'''
Given the root of a binary tree, invert the tree, and return its root.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root:
            left = self.invertTree(root.left)
            right = self.invertTree(root.right)
            root.left = right
            root.right = left
        return root
    


'''
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        else:
            return max(self.maxDepth(root.left) + 1, self.maxDepth(root.right) + 1)



'''
Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.

'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # longest left + longest right
        diameter = 0
        def dfs(node):
            if not node:
                return 0
            nonlocal diameter
            left = dfs(node.left)
            right = dfs(node.right)
            diameter = max(diameter, left + right)
            return max(left, right) + 1
         
        dfs(root)
        return diameter



'''
Given a binary tree, determine if it is
height-balanced
.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def getHeight(self, node):
        if node:
            return max(self.getHeight(node.left), self.getHeight(node.right)) + 1
        else:
            return 0
    def isBalanced(self, root: TreeNode) -> bool:       
        if not root:
            return True
        else:
            main = abs(self.getHeight(root.left) - self.getHeight(root.right)) <= 1 
            return main and self.isBalanced(root.left) and self.isBalanced(root.right)

'''
Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.


'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if not p:
            return False
        if not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
       
    
'''
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, s, t):
        nodes = []
        def dfs(root):
            nonlocal nodes
            if not root:
                return
            if root.val == t.val:
                nodes.append(root)
            dfs(root.left)
            dfs(root.right)

        def isSameTree(s, t):
            if not s and not t:
                return True
            if not s:
                return False
            if not t:
                return False
            if t.val != s.val:
                return False
            else:
                checkChildren = isSameTree(s.left, t.left) and isSameTree(s.right, t.right)
                return checkChildren
        
        if not t:
            return True
        dfs(s)
        for node in nodes:
            isSub = isSameTree(node, t)
            if isSub:
                return True
        return False
            
'''
Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return root
        if not p or not q:
            return p if p else q
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        # below can be replaced by else: return root
        elif p.val > root.val and q.val < root.val:
            return root
        elif q.val > root.val and p.val < root.val:
            return root
        elif q.val == root.val or p.val == root.val:
            return root
        else:
            return None

'''
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
'''
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        def bfs(root):
            res = []
            q = [(root, 0)]
            while q:
                node, level = q.pop(0)
                if node:
                    while len(res) <= level:
                        res.append([])
                    res[level].append(node.val)
                    q.append((node.left, level+1))
                    q.append((node.right, level+1))
            return res
        
        ret = bfs(root)
        return ret

'''
Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
'''

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        
        rightside = []
        
        def helper(node: TreeNode, level: int) -> None:
            if level == len(rightside):
                rightside.append(node.val)
            for child in [node.right, node.left]:
                if child:
                    helper(child, level + 1)
                
        helper(root, 0)
        return rightside


'''
Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.

'''



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        good = 0
        def dfs(root, seen):
            nonlocal good
            if not root:
                return
            if seen <= root.val:
                good += 1
            new_seen = max(root.val, seen)
            dfs(root.left, new_seen)
            dfs(root.right, new_seen)

        dfs(root, float('-inf'))
        return good
            


            

