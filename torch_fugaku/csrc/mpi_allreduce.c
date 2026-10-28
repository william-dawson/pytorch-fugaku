// Gradient allreduce over MPI for torch_fugaku's DDP communication hook,
// loaded through ctypes. Built against Open MPI 4.0.1, the base of Fujitsu
// MPI, so the same binary runs on Fujitsu MPI on Fugaku. Ranks come from
// mpiexec, so MPI rank order is the order torch's RANK uses (pf-run).
//
// Data goes through one reused 8 MB staging buffer. Without Fujitsu's large
// pages, Tofu must register every new buffer it sends, ~1.5 s per 25 MB
// bucket at 48 ranks, and torch's allocator keeps presenting new ones; a
// single small buffer is registered once.
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#define STAGE_FLOATS (2L << 20)  // 8 MB

static int world = 1;
static float *stage;

// Serialized threading: only one thread calls MPI at a time, but it need not
// be the thread that called MPI_Init (the autograd engine's may differ).
int omni_mpi_init(void) {
  int ok, provided;
  MPI_Initialized(&ok);
  if (!ok) MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  if (!stage && posix_memalign((void **)&stage, 1 << 21, STAGE_FLOATS * sizeof(float))) return -1;
  return world;
}

// In-place sum of n float32 values across all ranks; 0 on success.
int omni_mpi_sum_f32(float *buf, long n) {
  for (long off = 0; off < n; off += STAGE_FLOATS) {
    long m = n - off < STAGE_FLOATS ? n - off : STAGE_FLOATS;
    memcpy(stage, buf + off, m * sizeof(float));
    int rc = MPI_Allreduce(MPI_IN_PLACE, stage, (int)m, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) return rc;
    memcpy(buf + off, stage, m * sizeof(float));
  }
  return 0;
}

void omni_mpi_finalize(void) {
  int ok, done;
  MPI_Initialized(&ok);
  MPI_Finalized(&done);
  if (ok && !done) MPI_Finalize();
}
