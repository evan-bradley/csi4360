#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>

#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <stdint.h>

#define MUTEX_COUNT 16

struct node_s {
  uint8_t key[4][4];
  unsigned value;
  struct node_s *next;
};

typedef struct node_s node_t;

struct hashtable_s {
  unsigned int size;
  unsigned int bins;
  pthread_mutex_t **locks;
  node_t **table;
};

typedef struct hashtable_s hashtable_t;

struct kv_s {
  char **keys;
  char **values;
};

typedef struct kv_s kv_t;

struct ht_arg_s {
  hashtable_t *hashtable;
  kv_t *input;
  unsigned input_len;
  kv_t *output;
  unsigned output_len;
  unsigned threads;
  void (*cb)(void*, int, void*);
  unsigned count;
  pthread_mutex_t count_mut;
  pthread_barrier_t barrier;
};

typedef struct ht_arg_s ht_arg_t;

/*typedef struct {
  kv_t *kv;

  } work_queue;*/

pthread_mutex_t mutexes[MUTEX_COUNT];

#include "heap.h"

typedef struct {
  uint8_t board[4][4];
  uint8_t pos[2];
  hashtable_t *table;
  node_t *queue;
  pthread_mutex_t qmut;
  unsigned qlen;
} puzzle_t;

typedef struct {
  puzzle_t *puzzle;
  uint8_t solved;
  pthread_barrier_t barrier;
} puzzle_arg_t;

#define INIT_BOARD {{ 1,  2,  3,  4}, \
                    { 5,  6,  7,  8}, \
                    { 9, 10, 11, 12}, \
                    {13, 14, 15, 16}}

void shuffle_board(puzzle_t *puzzle);
void clone_board(uint8_t new_board[4][4], uint8_t board[4][4]);
int brdcmp(uint8_t board1[4][4], uint8_t board2[4][4]);
