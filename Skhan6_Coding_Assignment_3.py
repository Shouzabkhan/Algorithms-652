import sys

# Add your name and blazer id here
name = "shouzab khan"
blazer_id = "skhan6"

# For all questions, please do not hard-code your algorithms for testing inputs.
# You code should work with any input with arbitrary sizes. If you hard-code your algorithms, 10 points
# will be deduced per question.



# Question 1 (50 points): Implement a priority queue data structure using max-heap.
# Do not modify the given function signatures. You are free to add helper functions or variables inside class.

class PriorityQueue:
    def __init__(self, maximum_size):
        # Initialize an empty list for the heap and set the maximum size.
        self.data = []
        self.maximum_size = maximum_size

    # Helper function to maintain the max-heap property when moving elements down the heap.
    def _heapify_down(self, index):
        largest = index  # Assume the current index is the largest.
        left = 2 * index + 1  # Index of the left child.
        right = 2 * index + 2  # Index of the right child.

        # Check if the left child exists and is greater than the current largest.
        if left < len(self.data) and self.data[left] > self.data[largest]:
            largest = left
        # Check if the right child exists and is greater than the current largest.
        if right < len(self.data) and self.data[right] > self.data[largest]:
            largest = right

        # If the largest is not the current index, swap and recursively heapify.
        if largest != index:
            self.data[index], self.data[largest] = self.data[largest], self.data[index]
            self._heapify_down(largest)

    # Helper function to maintain the max-heap property when moving elements up the heap.
    def _heapify_up(self, index):
        while index > 0:  # Continue until reaching the root.
            parent = (index - 1) // 2  # Find the parent index.
            if self.data[index] > self.data[parent]:  # If the current node is greater than the parent, swap.
                self.data[index], self.data[parent] = self.data[parent], self.data[index]
                index = parent  # Move up to the parent index.
            else:
                break

    # Build a priority queue from a given list of elements.
    def BuildQueue(self, input):
        self.data = input  # Initialize the heap with the input list.
        # Heapify from the last non-leaf node to the root.
        for i in range(len(self.data) // 2 - 1, -1, -1):
            self._heapify_down(i)

    # Add an element to the priority queue.
    def Enqueue(self, x):
        if len(self.data) >= self.maximum_size:  # Check if the heap has reached its maximum size.
            return False
        self.data.append(x)  # Add the new element at the end of the heap.
        self._heapify_up(len(self.data) - 1)  # Restore the heap property.
        return True

    # Remove and return the largest element (highest priority).
    def Dequeue(self):
        if not self.data:  # Return 0 if the heap is empty.
            return 0
        max_value = self.data[0]  # The root is the maximum value.
        self.data[0] = self.data[-1]  # Replace the root with the last element.
        self.data.pop()  # Remove the last element.
        self._heapify_down(0)  # Restore the heap property.
        return max_value

# Question 2 (70 points): Implement a binary search tree data structure
# Do not modify the given function signatures. You are free to add helper functions or variables inside the class.

class TreeNode:
    def __init__(self, key):
        self.key = key
        self.parent = None  # Pointer to the parent node.
        self.leftchild = None  # Pointer to the left child.
        self.rightchild = None  # Pointer to the right child.

class BinarySearchTree:
    def __init__(self):
        self.root = None  # Initialize an empty tree.

    # Insert a key into the BST.
    def Insert(self, x):
        new_node = TreeNode(x)
        if self.root is None:  # If the tree is empty, the new node becomes the root.
            self.root = new_node
            return
        current = self.root
        while True:  # Traverse the tree to find the correct position for the new node.
            if x < current.key:  # Move to the left subtree.
                if current.leftchild is None:
                    current.leftchild = new_node
                    new_node.parent = current
                    break
                current = current.leftchild
            else:  # Move to the right subtree.
                if current.rightchild is None:
                    current.rightchild = new_node
                    new_node.parent = current
                    break
                current = current.rightchild

    # Search for a key in the BST.
    def Search(self, x):
        current = self.root
        while current is not None:  # Traverse the tree until the key is found or the end is reached.
            if current.key == x:
                return True
            elif x < current.key:  # Move to the left subtree.
                current = current.leftchild
            else:  # Move to the right subtree.
                current = current.rightchild
        return False

    # Delete a key from the BST.
    def Delete(self, x):
        node = self.root
        while node:  # Traverse to find the node to delete.
            if node.key == x:
                if not node.leftchild:  # Case: Node has no left child.
                    self._transplant(node, node.rightchild)
                elif not node.rightchild:  # Case: Node has no right child.
                    self._transplant(node, node.leftchild)
                else:  # Case: Node has two children.
                    successor = self._minimum(node.rightchild)
                    if successor.parent != node:
                        self._transplant(successor, successor.rightchild)
                        successor.rightchild = node.rightchild
                        successor.rightchild.parent = successor
                    self._transplant(node, successor)
                    successor.leftchild = node.leftchild
                    successor.leftchild.parent = successor
                return
            elif x < node.key:  # Move to the left subtree.
                node = node.leftchild
            else:  # Move to the right subtree.
                node = node.rightchild

    # Helper function to replace one subtree with another.
    def _transplant(self, u, v):
        if u.parent is None:  # If replacing the root.
            self.root = v
        elif u == u.parent.leftchild:  # If u is a left child.
            u.parent.leftchild = v
        else:  # If u is a right child.
            u.parent.rightchild = v
        if v:  # Update the parent pointer of v.
            v.parent = u.parent

    # Find the minimum node in a subtree.
    def _minimum(self, node):
        while node.leftchild:  # Traverse to the leftmost node.
            node = node.leftchild
        return node

    # Find and return the minimum key in the BST.
    def Minimum(self):
        current = self.root
        while current and current.leftchild:
            current = current.leftchild
        return current.key if current else 0

    # Find and return the maximum key in the BST.
    def Maximum(self):
        current = self.root
        while current and current.rightchild:
            current = current.rightchild
        return current.key if current else 0

    # Find and return the successor of a key in the BST.
    def Successor(self, x):
        current = self.root
        successor = None
        while current:
            if x < current.key:  # Potential successor.
                successor = current
                current = current.leftchild
            else:
                current = current.rightchild
        return successor.key if successor else -1

    # Find and return the predecessor of a key in the BST.
    def Predcessor(self, x):
        current = self.root
        predecessor = None
        while current:
            if x > current.key:  # Potential predecessor.
                predecessor = current
                current = current.rightchild
            else:
                current = current.leftchild
        return predecessor.key if predecessor else -1



#************ Do not modify code below this line ***************

def verify_array(a, b, size):
  if (len(a) < size or len(b) < size):
    return False
  for i in range(0, size):
    if (a[i] != b[i]):
     return False
  return True

def subtree_max_value(root):
  if (root == None):
    return -sys.maxsize - 1
  children_max = max(subtree_max_value(root.leftchild), subtree_max_value(root.rightchild))
  return max(root.key, children_max)

def subtree_min_value(root):
  if (root == None):
    return sys.maxsize
  children_min = min(subtree_min_value(root.leftchild), subtree_min_value(root.rightchild))
  return min(root.key, children_min)

def verify_tree(root):
  if (root == None):
    return True
  if (root.leftchild != None and root.key <= subtree_max_value(root.leftchild)):
    return False
  if (root.rightchild != None and root.key >= subtree_min_value(root.rightchild)):
    return False
  return verify_tree(root.leftchild) and verify_tree(root.rightchild)

result = [16, 15, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

print("Name: {} blazer_id: {}".format(name, blazer_id))

print("---------- Priority Queue ---------\n")
pq = PriorityQueue(15)
pq.BuildQueue([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
pass_check = verify_array(pq.data, [16, 14, 10, 8, 7, 9, 3, 2, 4, 1], 10)
print(f"BuildQueue: {'Pass' if pass_check else 'No Pass'}")
pq.Enqueue(12)
pass_check = verify_array(pq.data, [16, 14, 10, 8, 12, 9, 3, 2, 4, 1, 7], 11)
print(f"Enqueue(12): {'Pass' if pass_check else 'No Pass'}")
pq.Enqueue(5)
pass_check = verify_array(pq.data, [16, 14, 10, 8, 12, 9, 3, 2, 4, 1, 7, 5], 12)
print(f"Enqueue(5): {'Pass' if pass_check else 'No Pass'}")
pq.Enqueue(15)
pass_check = verify_array(pq.data, [16, 14, 15, 8, 12, 10, 3, 2, 4, 1, 7, 5, 9], 13)
print(f"Enqueue(15): {'Pass' if pass_check else 'No Pass'}")
pq.Enqueue(6)
pass_check = verify_array(pq.data, [16, 14, 15, 8, 12, 10, 6, 2, 4, 1, 7, 5, 9, 3], 14)
print(f"Enqueue(6): {'Pass' if pass_check else 'No Pass'}")
pq.Enqueue(11)
pass_check = verify_array(pq.data, [16, 14, 15, 8, 12, 10, 11, 2, 4, 1, 7, 5, 9, 3, 6], 15)
print(f"Enqueue(11): {'Pass' if pass_check else 'No Pass'}")
for i in range(0, 15):
  x = pq.Dequeue()
  pass_check = x == result[i]
  print(f"Dequeue current max  {'Pass' if pass_check else 'No Pass'}")

print("---------- Binary Search Tree ---------\n")
bst = BinarySearchTree()
input = [5, 3, 8, 1, 4, 6, 9, 2, 0, 7, 10, 11, 12]
for i in range(0, len(input)):
  bst.Insert(input[i])
  pass_check = verify_tree(bst.root)
  print("Insert key {}: {}".format(input[i], 'Pass' if pass_check else 'No Pass'))

for i in range(0, len(input)):
  pass_check = bst.Search(input[i])
  print("Search key {}: {}".format(input[i], 'Pass' if pass_check else 'No Pass'))

for i in range(13, 20):
  pass_check = not bst.Search(i)
  print("Search key {}: {}".format(i, 'Pass' if pass_check else 'No Pass'))

pass_check = bst.Minimum() == 0
print(f"Minimum key: {'Pass' if pass_check else 'No Pass'}")
pass_check = bst.Maximum() == 12
print(f"Maximum key: {'Pass' if pass_check else 'No Pass'}")

for i in range(0, len(input)-1):
  pass_check = bst.Successor(input[i]) == input[i]+1
  print("Successor of key {}: {}".format(input[i], 'Pass' if pass_check else 'No Pass'))

for i in range(1, len(input)):
  pass_check = bst.Predcessor(input[i]) == input[i]-1
  print("Predcessor of key {}: {}".format(input[i], 'Pass' if pass_check else 'No Pass'))

for i in range(0, len(input)):
  bst.Delete(input[i])
  pass_check = not bst.Search(input[i])
  print("Delete key {}: {}".format(input[i], 'Pass' if pass_check else 'No Pass'))



 # References for Priority Queue (Max-Heap):
# 1. Python Documentation - `heapq` Module: https://docs.python.org/3/library/heapq.html
# 2. CLRS (Introduction to Algorithms), Chapter 6: Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
# 3. GeeksforGeeks - Max Heap Implementation: https://www.geeksforgeeks.org/max-heap-in-python/

# References for Binary Search Tree (BST):
# 1. GeeksforGeeks - Binary Search Tree Operations: https://www.geeksforgeeks.org/binary-search-tree-data-structure/
# 2. Real Python - Binary Search Trees: https://realpython.com/python-data-structures/#binary-search-trees
# 4. CLRS (Introduction to Algorithms), Chapter 12: Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
