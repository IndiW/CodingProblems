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
            


'''
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

    The left
    subtree
    of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.


'''  

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def validate(node, low=-math.inf, high=math.inf):
            if not node:
                return True
            if node.val <= low or node.val >= high:
                return False
            
            return validate(node.right, node.val, high) and validate(node.left, low, node.val)
        
        return validate(root)
                

'''
Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

The following solution is an inorder traversal of the tree
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        def toArr(root):
            if not root:
                return []
            else:
                return toArr(root.left) + [root] + toArr(root.right)
        
        arr = toArr(root)
        return arr[k-1].val
    
# Iterative solution 

class Solution:
    def kthSmallest(self, root, k):
        stack = []
        
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop() #get last element ie top of stack
            k -= 1
            if not k:
                return root.val
            root = root.right




'''
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # preorder = node, left, right
        # inorder = left, node, right
        # for inorder, all elements to the left of the root val is in left subtree
        # the next element in inorder is the next root
        inorder_index_map = {}
        for i, nodeVal in enumerate(inorder):
            inorder_index_map[nodeVal] = i

        preorder_index = 0
        

        def arr_to_tree(left, right):
            nonlocal preorder_index
            if left > right:
                return None
            
            root_val = preorder[preorder_index]
            root = TreeNode(root_val)

            preorder_index += 1

            left_root_val = inorder_index_map[root_val] - 1
            right_root_val = inorder_index_map[root_val] + 1

            root.left = arr_to_tree(left, left_root_val)
            root.right = arr_to_tree(right_root_val, right)
            return root

        return arr_to_tree(0, len(preorder) - 1)


'''
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

'''
        
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        
        # scenarios:
        # Path goes from left to root to right
        # Path is in left
        # Path is in right
        # Path is root + left
        # Path is root + right
        # Path is just root


        # if node children are null, the mathPath is the node.val
        # get maxPathSum of left
        # get maxPathSum of right
        # we want to update the maxPath as we try all possibilities
        # when we return the maxPath of a subtree, we can't add both children so we have to choose the max child
        # if children are negative, we don't want to include them

        res = [root.val]

        def dfs(root):
            if not root:
                return 0
            leftMax = dfs(root.left)
            rightMax = dfs(root.right)

            # ignore negatives
            leftMax = max(leftMax, 0)
            rightMax = max(rightMax, 0)

            # max sum with split
            res[0] = max(res[0],root.val + leftMax + rightMax)
            # max sum without split
            return root.val + max(leftMax, rightMax)

        dfs(root)
        return res[0]



'''
Given a binary tree, return the inorder traversal
'''

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
        

                






