## Basic algoritms with AlgoExpoert

### Easy
- ###### [Two number sum in array](#two_number_sum)
- ###### [Find closest value in binary search tree](#find_closest_value_in_bst)
- ###### [Depth first search](#depth_first_search)
- ###### [Linked List Construction](#linked_list_construction)
- ###### [Nth fibonacci](#nth_fibonacci_algo)
- ###### [Find Three largest numbers](#find_three_largest_numbers)
- ###### [Binary search](#binary_search_algo)
- ###### [Insertion sort](#insertion_sort_algo)


#### two_number_sum

task: [-4, -1, 1, 3, 5, 6, 8, 11], 10
```python
# O(n) time | O(n) space
def two_number_sum(array, target_sum):
    nums = {}
    for num in array:
        potential_match = target_sum - num
        if potential_match in nums:
            return [potential_match, num]
        else:
            nums[num] = True
    return []

# O(nlog(n)) | O(1) space
def two_number_sum2(array, target_sum):
    array.sort()
    left = 0
    right = len(array) - 1
    while left < right:
        current_sum = array[left] + array[right]
        if current_sum == target_sum:
            return [array[left], array[right]]
        elif current_sum < target_sum:
            left += 1
        elif current_sum > target_sum:
            right -= 1
    return []
```

#### find_closest_value_in_bst
![](src/algos_bst_1.png)


```python
# average: O(Log(n)) time | O(Log(n)) space
# Worst: O(n) time | O(n) space

def find_closest_value_in_bst(tree, target):
    return find_closest_value_in_bst_helper(tree, target, float("inf"))

def find_closest_value_in_bst_helper(tree, target, closest):
    if tree is None:
        return closest
    if abs(target - closest) > abs(target - tree.value):
        closest = tree.value
    if target < tree.value:
        return find_closest_value_in_bst_helper(tree.left, target, closest)
    elif target > tree.value:
        return find_closest_value_in_bst_helper(tree.right, target, closest)
    else:
        return closest
```

```python
# average: O(Log(n)) time | O(Log(1)) space
# Worst: O(n) time | O(1) space

def find_closest_value_in_bst(tree, target):
    return find_closest_value_in_bst_helper(tree, target, float("inf"))

def find_closest_value_in_bst_helper(tree, target, closest):
    current_node = tree
    while current_node is not None:
        if abs(target - closest) > abs(target - current_node.value):
            closest = current_node.value
        if target < current_node.value:
            current_node = current_node.left
        elif target > current_node.value:
            current_node = current_node.right
        else:
            break
    return closest
```

#### depth_first_search
![](src/algos_depth_first_search.png)

```python
class Node:
    def __init__(self, name):
        self.children = []
        self.name = name

    def add_child(self, name):
        self.children.append(Node(name))

    # O(v + e) time | O(v) space
    def depth_first_search(self, array):
        array.append(self.name)
        for child in self.children:
            child.depth_first_search(array)
        return array
```


#### linked_list_construction
![](src/algos_linked_list.png)

```python
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    # O(1) time | O(1) space
    def set_head(self, node):
        if self.head is None:
            self.head = node
            self.tail = node
            return
        self.insert_before(self.head, node)

    # O(1) time | O(1) space
    def set_tail(self, node):
        if self.tail is None:
            self.set_head(node)
            return
        self.insert_after(self.tail, node)

    # O(1) time | O(1) space
    def insert_before(self, node, node_to_insert):
        if node_to_insert == self.head and node_to_insert == self.tail:
            return
        self.remove(node_to_insert)
        node_to_insert.prev = node.prev
        node_to_insert.next = node
        if node.prev is None:
            self.head = node_to_insert
        else:
            node.prev.next = node_to_insert
        node.prev = node_to_insert

    # O(1) time | O(1) space
    def insert_after(self, node, node_to_insert):
        if node_to_insert == self.head and node_to_insert == self.tail:
            return
        self.remove(node_to_insert)
        node_to_insert.prev = node
        node_to_insert.next = node.next
        if node.next is None:
            self.tail = node_to_insert
        else:
            node.next.prev = node_to_insert
        node.next = node_to_insert

    # O(p) time | O(1) space
    def insert_at_position(self, position, node_to_insert):
        if position == 1:
            self.set_head(node_to_insert)
            return 
        node = self.head
        current_position = 1
        while node is not None and current_position != position:
            node = node.next
            current_position += 1
        if node is not None:
            self.insert_before(node, node_to_insert)
        else:
            self.set_tail(node_to_insert)

    # O(n) time | O(1) space
    def remove_nodes_with_value(self, value):
        node = self.head
        while node is not None:
            node_to_remove = node
            node = node.next
            if node_to_remove.value == value:
                self.remove(node_to_remove)
    # O(1) time | O(1) space
    def remove(self, node):
        if node == self.head:
            self.head = self.head.next
        if node == self.tail:
            self.tail = self.tail.prev
        # N <--- 1 ----> N
        self.remove_nodes_bindings(node)
    
    # O(n) time | O(1) space
    def contains_node_with_value(self, value):
        node = self.head
        while node is not None and node.value != value:
            node = node.next
        return node is not None

    def remove_nodes_bindings(self, node):
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = None
        node.next = None
```

#### nth_fibonacci_algo
![](src/algos_nth_fibonacci.png)

![](src/algos_nth_fibonacci_2.png)

```python
# O(2^n) time | O(n) space
def get_nth_fib(n):
    if n == 2:
        return 1
    elif n == 1:
        return 0
    else:
        return get_nth_fib(n - 1) + get_nth_fib(n - 2)
```

```python
# O(n) time | O(n) space
def get_nth_fib(n, memoize={1: 0, 2: 1}):
    if n in memoize:
        return memoize[n]
    else:
        memoize[n] = get_nth_fib(n - 1, memoize) + get_nth_fib(n - 2, memoize)
        return memoize
```

```python
# O(n) time | O(1) space
def get_nth_fib(n):
    last_two = [0, 1]
    counter = 3
    while counter <= n:
        next_fib = last_two[0] + last_two[1]
        last_two[0] = last_two[1]
        last_two[1] = next_fib
        counter += 1
    return last_two[1] if n > 1 else last_two[0]
```

#### find_three_largest_numbers
```python
# O(n) time | O(1)
def find_three_largest_numbers(array):
    three_largest = [None, None, None]
    for num in array:
        update_largest(three_largest, num)
    return three_largest

def update_largest(three_largest, num):
    if three_largest[2] is None or num > three_largest[2]:
        shift_and_update(three_largest, num, 2)
    elif three_largest[1] is None or num > three_largest[1]:
        shift_and_update(three_largest, num, 1)
    elif three_largest[0] is None or num > three_largest[0]:
        shift_and_update(three_largest, num, 0)

def shift_and_update(array, num, idx):
    for i in range(idx + 1):
        if i == idx:
            array[i] = num
        else:
            array[i] = array[i + 1]
```


#### binary_search_algo

```python
# O(log(n)) time | O(log(n)) space
def binary_search(array, target):
    return binary_search_helper(array, target, 0, len(array) - 1)

def binary_search_helper(array, target, left, right):
    if left > right:
        return -1
    middle = (left + right) // 2
    potential_match = array[middle]
    if target == potential_match:
        return middle
    elif target < potential_match:
        return binary_search_helper(array, target, left, middle - 1)
    else:
        return binary_search_helper(array, target, middle + 1, right)
```
```python
# O(log(n)) time | O(1) space
def binary_search(array, target):
    return binary_search_helper(array, target, 0, len(array) - 1)

def binary_search_helper(array, target, left, right):
    while left <= right:
        middle = (left + right) // 2
        potential_match = array[middle]
        if target == potential_match:
            return middle
        elif target < potential_match:
            right = middle - 1
        else:
            left = middle + 1
    return -1
```

#### insertion_sort_algo

```python
# O(n^2) time | O(1) space
def insertion_sort(array):
    for i in range(1, len(array)):
        j = i
        while j > 0 and array[j] < array[j - 1]:
            swap(j, j - 1, array)
            j -= 1

def swap(i, j, array):
    array[i], array[j] = array[j], array[i]
```