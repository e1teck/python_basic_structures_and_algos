
## Basics data structures

- ###### [Linked List](#linked_list)



#### linked_list
![](src/linked_list.png)

#### implementation

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