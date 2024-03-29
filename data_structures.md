
## Basics data structures

- ###### [Linked List](#linked_list)
- ###### [Queue](#standard_queue)
- ###### [Stack](#standard_stack)
- ###### [Binary Tree](#binary_tree)

#### linked_list
![](src/linked_list.png)

###### implementation

```python
class Node:
    """
    This Node class has been created for you.
    It contains the necessary properties for the solution, which are:
    - name
    - matric
    - year
    - next
    """

    def __init__(self, name, matric, year):
        self.name = name
        self.matric = matric
        self.year = year
        self.__next = None

    def set_next(self, node):
        if isinstance(node, Node) or node is None:
            self.__next = node
        else:
            raise TypeError("The 'next' node must be of type Node or None.")

    def get_next(self):
        return self.__next

    def print_details(self):
        print("{}: {} (year {})".format(self.matric, self.name, self.year))


class LinkedList:
    """
    This class is the one you should be modifying!
    Don't change the name of the class or any of the methods.

    Implement those methods that current raise a NotImplementedError
    """
    def __init__(self):
        self.__root = None

    def get_root(self):
        return self.__root

    def add_to_list(self, node):
        if self.__root:
            node.set_next(self.__root)
        self.__root = node

    def print_list(self):
        marker = self.__root
        while marker:
            marker.print_details()
            marker = marker.get_next()

    def find(self, name):
        marker = self.__root
        while marker:
            if marker.name == name:
                return marker
            marker = marker.get_next()
        raise LookupError("Name {} was not found in the linked list.".format(name))
        
        
if __name__ == '__main___':
    names = ("Jose", "1234", 2), ("Rolf", "2345", 3), ("Anna", "3456", 7)
    nodes = [Node(name, matric, year) for name, matric, year in names]
    linked_list = LinkedList()
    
    for node in nodes:
        linked_list.add_to_list(node)
    
    marker = linked_list.get_root()
    for i in range(len(nodes)-1, -1, -1):
        print(marker.name)
        marker = marker.get_next()
```


#### standard_queue
![](src/queue.png)
###### implementation

```python
class LinkedList:
    """
    You should implement the methods of this class which are currently
    raising a NotImplementedError!
    Don't change the name of the class or any of the methods.
    """
    def __init__(self):
        self.__root = None

    def get_root(self):
        return self.__root

    def add_start_to_list(self, node):
        if self.__root:
            node.set_next(self.__root)
        self.__root = node

    def remove_end_from_list(self):
        marker = self.__root

        # Especially delete the root if it by itself.
        if not marker.get_next():
            self.__root = None
            return marker

        # Iterate over each Node in this list
        while marker:
            # Get the next node
            following_node = marker.get_next()
            if following_node:
                # If the next Node's next Node is None, it means the current marker is the
                # second-to-last Node (there is only one more after it).
                if not following_node.get_next():
                    # Make the marker's next = None so the very last Node is removed.
                    marker.set_next(None)
                    return following_node
            marker = marker.get_next()

    def print_list(self):
        marker = self.__root
        while marker:
            marker.print_details()
            marker = marker.get_next()

    def find(self, name):
        marker = self.__root
        while marker:
            if marker.name == name:
                return marker
            marker = marker.get_next()
        raise LookupError("Name {} not found in the linked list.".format(name))

    def size(self):
        marker = self.__root
        count = 0
        while marker:
            count += 1
            marker = marker.get_next()
        return count
        
        

class LinkedQueue:
    """
    This class is a queue wrapper around a LinkedList.

    This means that methods like `add_to_list_start` should now be called `push`, for example.

    Don't modify class or method names, just implement methods that currently raise
    a NotImplementedError!
    """

    def __init__(self):
        self.__linked_list = LinkedList()

    def push(self, node):
        self.__linked_list.add_start_to_list(node)

    def pop(self):
        return self.__linked_list.remove_end_from_list()

    def find(self, name):
        return self.__linked_list.find(name)

    def print_details(self):
        self.__linked_list.print_list()

    def __len__(self):
        return self.__linked_list.size()


class Node:
    """
    This Node class has been created for you.
    It contains the necessary properties for the solution, which are:
    - name
    - phone
    - next
    """

    def __init__(self, name, phone):
        self.name = name
        self.phone = phone
        self.__next = None

    def set_next(self, node):
        if isinstance(node, Node) or node is None:
            self.__next = node
        else:
            raise TypeError("The 'next' node must be of type Node or None.")

    def get_next(self):
        return self.__next

    def print_details(self):
        print("{} ({})".format(self.name, self.phone))
        
if __name__ == '__main___':
    names = ("Jose", "1234-356"), ("Rolf", "2345-1-53563-2"), ("Anna", "345623-16779-3")
    nodes = [Node(name, phone) for name, phone in names]
    linked_list = LinkedList()
    
    for node in nodes:
        linked_list.add_start_to_list(node)
    
    marker = linked_list.get_root()
    for i in range(len(nodes)-1, -1, -1):
        print(marker.name)
        marker = marker.get_next()
    
    node = Node("Jose", "123-456-7890")
    queue = LinkedQueue()
    
    queue.push(node)
    print(len(queue))
    
    popped = queue.pop()
    
    print(popped.name)
    print(len(queue))

```

#### standard_stack
![](src/stack.png)

###### implementation

```python
class LinkedList:
    """
    You should implement the methods of this class which are currently
    raising a NotImplementedError!
    Don't change the name of the class or any of the methods.
    """
    def __init__(self):
        self.__root = None

    def get_root(self):
        return self.__root

    def add_start_to_list(self, node):
        if self.__root:
            node.set_next(self.__root)
        self.__root = node

    def remove_start_from_list(self):
        if not self.__root:
            raise RuntimeError("Tried to remove from the list but it was already empty!")
        removed_node = self.__root
        self.__root = self.__root.get_next()
        return removed_node

    def print_list(self):
        marker = self.__root
        while marker:
            marker.print_details()
            marker = marker.get_next()

    def find(self, text):
        marker = self.__root
        while marker:
            if marker.text == text:
                return marker
            marker = marker.get_next()
        raise LookupError("Node with text {} was not found in the linked list!".format(text))

    def size(self):
        marker = self.__root
        count = 0
        while marker:
            count += 1
            marker = marker.get_next()
        return count


class LinkedStack:
    """
    This class is a stack wrapper around a LinkedList.

    This means that methods like `add_to_list_start` should now be called `push`, for example.

    Don't modify class or method names, just implement methods that currently raise
    a NotImplementedError!
    """

    def __init__(self):
        self.__linked_list = LinkedList()

    def push(self, node):
        self.__linked_list.add_start_to_list(node)

    def pop(self):
        return self.__linked_list.remove_start_from_list()

    def print_details(self):
        self.__linked_list.print_list()

    def __len__(self):
        return self.__linked_list.size()


class Node:
    """
    This Node class has been created for you.
    It contains the necessary properties for the solution, which are:
    - text
    - next
    """

    def __init__(self, text):
        self.text = text
        self.__next = None

    def set_next(self, node):
        if isinstance(node, Node) or node is None:
            self.__next = node
        else:
            raise TypeError("The 'next' node must be of type Node or None.")

    def get_next(self):
        return self.__next

    def print_details(self):
        print("{}".format(self.text))
        

if __name__ == '__main__':
    names = ("Jose", "Rolf", "Anna")
    nodes = [Node(name) for name in names]

    linked_list = LinkedList()

    for node in nodes:
        linked_list.add_start_to_list(node)

    marker = linked_list.get_root()
    for i in range(len(nodes)-1, -1, -1):
        print(marker.text)
        marker = marker.get_next()
    
    some_node = linked_list.find("Anna")
    print('Found {}'.format(some_node.text if some_node else None))
    popped_node = linked_list.remove_start_from_list()
    print('Removed {}'.format(popped_node.text))
    
    try:
        some_node = linked_list.find("Anna")
        print('Found {}'.format(some_node.text if some_node else None))
    except Exception as _err:
        print(some_node.text, 'not found')
    
    name = "Jose"

    node = Node(name)
    stack = LinkedStack()

    stack.push(node)
    
    print(len(stack))
    popped = stack.pop()
    
    print(popped.text)
    
    print(len(stack))
    

```

#### binary_tree
![](src/binary_tree.png)

###### implementation

```python
class BinaryTree:
    """
    This class is a binary tree implementation.

    Don't modify class or method names, just implement methods that currently raise
    a NotImplementedError!
    """

    def __init__(self):
        self.__root = None

    def get_root(self):
        return self.__root

    def add(self, node):
        # The root is None, so set the root to be the new Node.
        if not self.__root:
            self.__root = node
        else:
            # Start iterating over the tree.
            marker = self.__root
            while marker:
                if node.value == marker.value:
                    raise ValueError("A node with that value already exists.")
                elif node.value > marker.value:
                    if not marker.get_right():
                        marker.set_right(node)
                        return
                    else:
                        marker = marker.get_right()
                else:
                    if not marker.get_left():
                        marker.set_left(node)
                        return
                    else:
                        marker = marker.get_left()

    def find(self, value):
        marker = self.__root
        while marker:
            if value == marker.value:
                return marker
            elif value > marker.value:
                marker = marker.get_right()
            else:
                marker = marker.get_left()
        raise LookupError("A node with value {} was not found.".format(value))

    def print_inorder(self):
        self.__print_inorder_r(self.__root)

    def __print_inorder_r(self, current_node):
        if not current_node:
            return
        self.__print_inorder_r(current_node.get_left())
        print(current_node.print_details())
        self.__print_inorder_r(current_node.get_right())


class Node:
    """
    This Node class has been created for you.
    It contains the necessary properties for the solution, which are:
    - text
    - next
    """

    def __init__(self, data, value):
        self.data = data
        self.value = value
        self.__left = None
        self.__right = None

    def set_right(self, node):
        if isinstance(node, Node) or node is None:
            self.__right = node
        else:
            raise TypeError("The 'right' node must be of type Node or None.")

    def set_left(self, node):
        if isinstance(node, Node) or node is None:
            self.__left = node
        else:
            raise TypeError("The 'left' node must be of type Node or None.")

    def get_right(self):
        return self.__right

    def get_left(self):
        return self.__left

    def print_details(self):
        print("{}: {}".format(self.value, self.data))


if __name__ == '__main__':
    names = (("Jose", 2), ("Rolf", 1), ("Anna", 3))
    nodes = [Node(name, value) for name, value in names]
    binary_tree = BinaryTree()

    for node in nodes:
        binary_tree.add(node)

    print(binary_tree.get_root().data)
    print(binary_tree.get_root().get_left().data)
    print(binary_tree.get_root().get_right().data)
    
    for i in range(0, len(nodes)):
        print(nodes[i].value)
        print(binary_tree.find(nodes[i].value).data)

```
