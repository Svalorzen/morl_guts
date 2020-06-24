import heapq


'''
    HEAPQ PRIORITYQUEUE IMPLEMENTATION
    Acts like Python's PriorityQueue implementation,
    but twice as fast. Adapted from
    http://stackoverflow.com/questions/407734/a-generic-priority-queue-for-python
'''


class PriorityQueueSet(object):

    """
    Combined priority queue and set data structure.

    Acts like a priority queue, except that its items are guaranteed to be
    unique. Provides O(1) membership test, O(log N) insertion and O(log N)
    removal of the smallest item.

    Important: the items of this data structure must be both comparable and
    hashable (i.e. must implement __cmp__ and __hash__). This is true of
    Python's built-in objects, but you should implement those methods if you
    want to use the data structure for custom objects.
    """

    def __init__(self, items=[]):
        """
        Create a new PriorityQueueSet.

        Arguments:
            items (list): An initial item list - it can be unsorted and
                non-unique. The data structure will be created in O(N).
        """
        self.set = dict((item, True) for item in items)
        self.heap = list(self.set.keys())
        heapq.heapify(self.heap)

    def has_item(self, item):
        """Check if ``item`` exists in the queue."""
        return item in self.set

    def empty(self):
        if self.heap:
            return False
        return True

    def get(self):
        """Remove and return the smallest item from the queue."""
        smallest = heapq.heappop(self.heap)
        del self.set[smallest]
        return smallest

    def remove_item(self, item):
        """
        """
        list_of_pops = []
        while not self.empty():
            smallest = heapq.heappop(self.heap)
            del self.set[smallest]
            print("smol, ", smallest)
            if smallest[1] != item:
                list_of_pops.append(smallest)
            else:
                print("Removing {}".format(smallest))
        for popped in list_of_pops:
            self.add(popped)

    def peak_all(self):
        """ Please don't hate me for this abomination """
        list_of_pops = []
        while not self.empty():
            smallest = heapq.heappop(self.heap)
            del self.set[smallest]
            list_of_pops.append(smallest)
        for popped in list_of_pops:
            self.add(popped)
        return list_of_pops

    def add(self, item):
        """Add ``item`` to the queue if doesn't already exist."""
        if item not in self.set:
            self.set[item] = True
            heapq.heappush(self.heap, item)
