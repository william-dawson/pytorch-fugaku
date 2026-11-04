// Can an allgather overlap with 12-thread OpenMP compute on Fugaku?
//
// For each per-rank message size, measures (max over ranks, median over reps):
//   comm      MPI_Iallgather + MPI_Wait, nothing else
//   compute   the OpenMP kernel alone, sized to take about as long as comm
//   and, communication and compute together, five ways:
//   iwait     MPI_Iallgather; compute; MPI_Wait
//   itest     same, but compute split into 16 chunks with MPI_Test between them
//   section   MPI_Iallgather; FJMPI_Progress_start; compute; FJMPI_Progress_stop; MPI_Wait
//             (only meaningful with opal_progress_thread_mode 1 or 2)
//   thread    helper std::thread runs blocking MPI_Allgather while the main thread computes
//             (what torch's ProcessGroupMPI does)
//   persist   MPIX_Allgather_init once, then MPI_Start; compute; MPI_Wait
//             (Tofu hardware offload from the second start, per the MPI manual 5.5.3)
// overlap% = (comm + compute - together) / min(comm, compute): 100 = fully hidden, 0 = serial.
//
// usage: mpiexec ... ./overlap <kernel: fma|stream> <reps> <bytes per rank>...
#include <mpi.h>
#include <mpi-ext.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int rank_, size_;
static std::string kernel;
static volatile float sink;
static std::vector<float> A, B, C;

static void work(long iters) {
  if (kernel == "fma") {
#pragma omp parallel
    {
      float x[1024];
      for (int i = 0; i < 1024; ++i) x[i] = i * 1e-3f;
      for (long it = 0; it < iters; ++it)
        for (int i = 0; i < 1024; ++i) x[i] = x[i] * 0.999f + 0.001f;
      if (x[7] == 12345.f) sink = x[7];
    }
  } else {
    const long n = static_cast<long>(A.size());
    for (long it = 0; it < iters; ++it) {
#pragma omp parallel for
      for (long i = 0; i < n; ++i) A[i] = B[i] + 0.5f * C[i];
    }
    sink = A[n / 2];
  }
}

static double max_over_ranks(double t) {
  double m;
  MPI_Allreduce(&t, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return m;
}

template <class F>
static double timed(F f, int reps) {
  std::vector<double> ts;
  for (int r = 0; r < reps + 1; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    f();
    double t = max_over_ranks(MPI_Wtime() - t0);
    if (r > 0) ts.push_back(t);  // first rep is warm-up
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

static bool check(const std::vector<char> &recv, size_t n) {
  for (int r = 0; r < size_; ++r)
    if (recv[r * n] != static_cast<char>(r + 1) || recv[r * n + n - 1] != static_cast<char>(r + 1)) return false;
  return true;
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  if (argc < 4) {
    if (rank_ == 0) std::fprintf(stderr, "usage: overlap fma|stream reps bytes...\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  kernel = argv[1];
  int reps = std::atoi(argv[2]);
  if (kernel == "stream") A.assign(8 << 20, 1.f), B.assign(8 << 20, 2.f), C.assign(8 << 20, 3.f);
  const char *mode = std::getenv("OMPI_MCA_opal_progress_thread_mode");
  if (rank_ == 0)
    std::printf("# ranks=%d threads=%d provided=%d kernel=%s progress_mode=%s\n", size_, omp_get_max_threads(),
                provided, kernel.c_str(), mode ? mode : "unset");
  for (int a = 3; a < argc; ++a) {
    size_t n = std::strtoull(argv[a], nullptr, 10);
    std::vector<char> send(n, static_cast<char>(rank_ + 1)), recv(n * size_);
    auto gather_nb = [&](MPI_Request *req) {
      MPI_Iallgather(send.data(), static_cast<int>(n), MPI_BYTE, recv.data(), static_cast<int>(n), MPI_BYTE,
                     MPI_COMM_WORLD, req);
    };
    double t_comm = timed([&] { MPI_Request q; gather_nb(&q); MPI_Wait(&q, MPI_STATUS_IGNORE); }, reps);

    // Size the kernel so compute ~= comm, where overlap matters most.
    long iters = 1;
    for (;;) {
      double t = timed([&] { work(iters); }, 1);
      if (t > t_comm / 4 || iters > (1L << 30)) {
        iters = std::max(1L, static_cast<long>(iters * t_comm / t));
        break;
      }
      iters *= 4;
    }
    MPI_Bcast(&iters, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    double t_comp = timed([&] { work(iters); }, reps);
    const long chunk = std::max(1L, iters / 16);

    struct Variant { const char *name; double t; bool ok; };
    std::vector<Variant> vs;
    auto run = [&](const char *name, auto f) {
      std::fill(recv.begin(), recv.end(), 0);
      double t = timed(f, reps);
      int ok = check(recv, n), all;
      MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
      vs.push_back({name, t, all != 0});
    };
    run("iwait", [&] { MPI_Request q; gather_nb(&q); work(iters); MPI_Wait(&q, MPI_STATUS_IGNORE); });
    run("itest", [&] {
      MPI_Request q;
      gather_nb(&q);
      int done = 0;
      for (long d = 0; d < iters; d += chunk) {
        work(std::min(chunk, iters - d));
        if (!done) MPI_Test(&q, &done, MPI_STATUS_IGNORE);
      }
      MPI_Wait(&q, MPI_STATUS_IGNORE);
    });
    run("section", [&] {
      MPI_Request q;
      gather_nb(&q);
      FJMPI_Progress_start();
      work(iters);
      FJMPI_Progress_stop();
      MPI_Wait(&q, MPI_STATUS_IGNORE);
    });
    run("thread", [&] {
      std::thread t([&] {
        MPI_Allgather(send.data(), static_cast<int>(n), MPI_BYTE, recv.data(), static_cast<int>(n), MPI_BYTE,
                      MPI_COMM_WORLD);
      });
      work(iters);
      t.join();
    });
    MPI_Request pq;
    MPIX_Allgather_init(send.data(), static_cast<int>(n), MPI_BYTE, recv.data(), static_cast<int>(n), MPI_BYTE,
                        MPI_COMM_WORLD, MPI_INFO_NULL, &pq);
    run("persist", [&] { MPI_Start(&pq); work(iters); MPI_Wait(&pq, MPI_STATUS_IGNORE); });
    MPI_Request_free(&pq);
    // The comm time for the persistent request alone, since Tofu offload may change it.
    MPIX_Allgather_init(send.data(), static_cast<int>(n), MPI_BYTE, recv.data(), static_cast<int>(n), MPI_BYTE,
                        MPI_COMM_WORLD, MPI_INFO_NULL, &pq);
    double t_pcomm = timed([&] { MPI_Start(&pq); MPI_Wait(&pq, MPI_STATUS_IGNORE); }, reps);
    MPI_Request_free(&pq);

    if (rank_ == 0) {
      for (auto &v : vs) {
        double c = std::strcmp(v.name, "persist") == 0 ? t_pcomm : t_comm;
        std::printf(
            "{\"kernel\":\"%s\",\"mode\":\"%s\",\"bytes\":%zu,\"variant\":\"%s\",\"comm_s\":%.6f,"
            "\"compute_s\":%.6f,\"together_s\":%.6f,\"overlap_pct\":%.1f,\"ok\":%d}\n",
            kernel.c_str(), mode ? mode : "unset", n, v.name, c, t_comp, v.t,
            100.0 * (c + t_comp - v.t) / std::min(c, t_comp), v.ok ? 1 : 0);
      }
      std::fflush(stdout);
    }
  }
  MPI_Finalize();
  return 0;
}
