def partition_by_confidence(arr, low, high):
    # Use the last element as the pivot (based on 'fingerprinted_confidence' and 'input_confidence')
    pivot_fingerprinted = arr[high]['fingerprinted_confidence']
    pivot_input = arr[high]['input_confidence']
    i = low - 1  # Pointer for the smaller element
    # Rearrange the array to put elements with higher confidence values to the left of the pivot
    for j in range(low, high):
        if (arr[j]['fingerprinted_confidence'] > pivot_fingerprinted or 
            (arr[j]['fingerprinted_confidence'] == pivot_fingerprinted and arr[j]['input_confidence'] > pivot_input)):
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # Swap if element has higher confidence than pivot
    # Swap the pivot element with the element at index i + 1
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    return i + 1

def quick_sort_by_confidence(arr):
    # Initialize the stack to simulate recursive calls
    stack = []
    # Push initial values of low and high to the stack
    stack.append((0, len(arr) - 1))
    # Keep processing until the stack is empty
    while stack:
        # Pop the top values (low, high) from the stack
        low, high = stack.pop()
        # Only sort if there's more than one element in the range
        if low < high:
            # Partition the array and get the pivot index
            pi = partition_by_confidence(arr, low, high)
            # Push the ranges for left and right sub-arrays onto the stack
            stack.append((low, pi - 1))  # Left part
            stack.append((pi + 1, high))  # Right part


def matches_partition(arr, low, high, key):
    # Choose the pivot (here we choose the last element)
    pivot = arr[high]
    i = low - 1  # Pointer for the "less than pivot" region
    for j in range(low, high):
        if key(arr[j]) <= key(pivot):
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    # Swap the pivot with the element at i+1 to place the pivot in the correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return i + 1

def matches_quick_sort(arr, key=lambda x: x):
    # Create a stack for the left and right indices
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            # Partition the array and get the pivot index
            pivot_index = matches_partition(arr, low, high, key)
            # Push the left and right sub-arrays onto the stack
            stack.append((low, pivot_index - 1))  # Left sub-array
            stack.append((pivot_index + 1, high))  # Right sub-array

    return arr


def quicksort_iterative_partition(arr, low, high, key, reverse):
    pivot = arr[high]  # Choose the last element as the pivot
    i = low - 1  # Pointer for the smaller element
    for j in range(low, high):
        if (key(arr[j]) <= key(pivot)) if not reverse else (key(arr[j]) >= key(pivot)):
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return i + 1


def quicksort_iterative(arr, key, reverse=True):
    # Stack to simulate recursion
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            # Partition step
            pivot_index = quicksort_iterative_partition(arr, low, high, key, reverse)
            # Push subarrays onto stack
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))
    
    return arr
