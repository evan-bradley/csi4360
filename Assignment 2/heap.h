/*
 * From: https://www.geeksforgeeks.org/k-ary-heap/
 * Author: Anurag Rai
 * Probably don't actually need a d-ary heap, but
 * I thought it would be a natural application,
 * since each board has at most 4 moves, which
 * would fit a 4-heap neatly.
 */

/*typedef struct {
  uint8_t board[4][4];
  unsigned value;
  } node_t;*/

#define LIMIT 10000

// function to heapify (or restore the min- heap
// property). This is used to build a k-ary heap
// and in extract_min()
// att[] -- Array that stores heap
// len   -- Size of array
// index -- index of element to be restored
//          (or heapified)
void restoreDown(node_t arr[], unsigned len, int index, int k) {
    // child array to store indexes of all
    // the children of given node
    int child[k+1];
 
    while (1)
    {
        // child[i]=-1 if the node is a leaf
        // children (no children)
        for (int i=1; i<=k; i++)
            child[i] = ((k*index + i) < len) ? (k*index + i) : -1;
 
        // min_child stores the minimum child and
        // min_child_index holds its index
        node_t min_child;
        min_child.value = LIMIT;

        int min_child_index ;
 
        // loop to find the minimum of all
        // the children of a given node
        for (int i=1; i<=k; i++) {
            if (child[i] != -1 && arr[child[i]].value < min_child.value) {
                min_child_index = child[i];
                min_child = arr[child[i]];
            }
        }
 
        // leaf node
        if (min_child.value == LIMIT)
            break;
 
        // swap only if the key of min_child_index
        // is less than the key of node
        if (arr[index].value > arr[min_child_index].value) {
          node_t temp = arr[index];
          arr[index] = arr[min_child_index];
          arr[min_child_index] = temp;
        }
 
        index = min_child_index;
    }
}
 
// Restores a given node up in the heap. This is used
// in decreaseKey() and insert()
void restoreUp(node_t arr[], int index, int k)
{
    // parent stores the index of the parent variable
    // of the node
    int parent = (index-1)/k;
 
    // Loop should only run till root node in case the
    // element inserted is the minimum restore up will
    // send it to the root node
    while (parent>=0)
    {
        if (arr[index].value < arr[parent].value)
        {
            node_t temp = arr[index];
            arr[index] = arr[parent];
            arr[parent] = temp;

            index = parent;
            parent = (index -1)/k;
        }
 
        // node has been restored at the correct position
        else
            break;
    }
}
 
// Function to build a heap of arr[0..n-1] and alue of k.
void buildHeap(node_t arr[], unsigned n, int k)
{
    // Heapify all internal nodes starting from last
    // non-leaf node all the way upto the root node
    // and calling restore down on each
    for (int i= (n-1)/k; i>=0; i--)
      restoreDown(arr, n, i, k);
}
 
// Function to insert a value in a heap. Parameters are
// the array, size of heap, value k and the element to
// be inserted
void insert(node_t arr[], unsigned* n, int k, node_t elem)
{
    // Put the new element in the last position
    arr[*n] = elem;
 
    // Increase heap size by 1
    *n = *n+1;
 
    // Call restoreUp on the last index
    restoreUp(arr, *n-1, k);
}
 
// Function that returns the key of root node of
// the heap and then restores the heap property
// of the remaining nodes
node_t extract_min(node_t arr[], unsigned* n, int k)
{
    // Stores the key of root node to be returned
    node_t min = arr[0];
 
    // Copy the last node's key to the root node
    arr[0] = arr[*n-1];
    printf("made it this far\n");
 
    // Decrease heap size by 1
    *n = *n-1;
 
    // Call restoreDown on the root node to restore
    // it to the correct position in the heap
    restoreDown(arr, *n, 0, k);
 
    return min;
}
