// Reduce-scatter on Fugaku: which MPI building blocks are fastest, and which keep MPI's memory
// flat as the rank count grows?
//
// One variant per process (MPI keeps per-peer buffers once a peer is contacted, so separate runs
// isolate each algorithm's footprint):
//   a2a_staged  fjmpi's current one: 8 MB staging chunks, MPI_Alltoall, local sum over ranks
//   a2a         the same, unstaged (one MPI_Alltoall)
//   rsb         MPI_Reduce_scatter_block
//   ring        N-1 rounds of MPI_Sendrecv with the neighbours (2 peers), add, pass on
//   halving     recursive halving (Rabenseifner): log2 N partner exchanges, power-of-two N only
//   allreduce   MPI_Allreduce of the whole buffer, then this rank's slice
// and the allgathers ZeRO-3 pairs with it (input: this rank's share, output: everyone's):
//   ag_staged   fjmpi's current one: 8 MB staging chunks of MPI_Allgather
//   ag          one MPI_Allgather on the buffers themselves
//   ag_ring     N-1 rounds of MPI_Sendrecv with the neighbours, straight into the output
//   ag_persist  MPIX_Allgather_init once per size (Tofu-offloaded), then MPI_Start/MPI_Wait
// Input on rank r element i is (r % 7) + (i % 5): sums are exact in float32, so the check is exact.
// Prints one JSON line per size on rank 0: time per call (median of reps, max over ranks) and node
// MemAvailable before MPI_Init and after the timed calls (min over nodes).
//
// usage: mpiexec ... ./rs_bench <variant> <reps> <floats per call>...
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" int MPIX_Allgather_init(const void *, int, MPI_Datatype, void *, int, MPI_Datatype, MPI_Comm, MPI_Info,
                                   MPI_Request *);

static int rank_, size_;

static double mem_avail_gb() {
  std::ifstream f("/proc/meminfo");
  std::string k;
  long v;
  std::string unit;
  while (f >> k >> v >> unit)
    if (k == "MemAvailable:") return v / 1048576.0;
  return -1;
}

static void add_into(float *acc, const float *x, size_t n) {
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) acc[i] += x[i];
}

// in: size_ blocks of n floats; out: n floats = sum over ranks of their block rank_.
static void a2a_staged(const float *in, float *out, size_t n, std::vector<float> &si, std::vector<float> &so) {
  const size_t chunk = (8u << 20) / (sizeof(float) * size_);  // fjmpi: kStage / (es * size)
  for (size_t off = 0; off < n; off += chunk) {
    const size_t cnt = std::min(chunk, n - off);
    for (int r = 0; r < size_; ++r) std::memcpy(&si[r * cnt], in + r * n + off, cnt * sizeof(float));
    MPI_Alltoall(si.data(), static_cast<int>(cnt), MPI_FLOAT, so.data(), static_cast<int>(cnt), MPI_FLOAT,
                 MPI_COMM_WORLD);
#pragma omp parallel for
    for (size_t j = 0; j < cnt; ++j) {
      float s = 0;
      for (int r = 0; r < size_; ++r) s += so[r * cnt + j];
      out[off + j] = s;
    }
  }
}

static void a2a(const float *in, float *out, size_t n, std::vector<float> &recv) {
  MPI_Alltoall(in, static_cast<int>(n), MPI_FLOAT, recv.data(), static_cast<int>(n), MPI_FLOAT, MPI_COMM_WORLD);
#pragma omp parallel for
  for (size_t j = 0; j < n; ++j) {
    float s = 0;
    for (int r = 0; r < size_; ++r) s += recv[r * n + j];
    out[j] = s;
  }
}

// Ring: acc starts as a copy of in; at round k send block (rank-k-1) to the next rank and add the
// block (rank-k-2) received from the previous one; after N-1 rounds block `rank` is complete.
static void ring(const float *in, float *out, size_t n, std::vector<float> &acc, std::vector<float> &tmp) {
  std::memcpy(acc.data(), in, n * size_ * sizeof(float));
  const int next = (rank_ + 1) % size_, prev = (rank_ - 1 + size_) % size_;
  for (int k = 0; k < size_ - 1; ++k) {
    const int s = ((rank_ - k - 1) % size_ + size_) % size_, r = ((rank_ - k - 2) % size_ + size_) % size_;
    MPI_Sendrecv(&acc[s * n], static_cast<int>(n), MPI_FLOAT, next, k, tmp.data(), static_cast<int>(n), MPI_FLOAT,
                 prev, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    add_into(&acc[r * n], tmp.data(), n);
  }
  std::memcpy(out, &acc[rank_ * n], n * sizeof(float));
}

// Recursive halving: keep the half of the current block range that holds this rank's block,
// exchange the other half with partner rank ^ d, add; ends with block `rank`.
static void halving(const float *in, float *out, size_t n, std::vector<float> &acc, std::vector<float> &tmp) {
  std::memcpy(acc.data(), in, n * size_ * sizeof(float));
  int lo = 0, hi = size_;
  for (int d = size_ / 2; d >= 1; d /= 2) {
    const int partner = rank_ ^ d, mid = (lo + hi) / 2;
    const bool upper = rank_ & d;
    const int keep_lo = upper ? mid : lo, send_lo = upper ? lo : mid;
    const size_t cnt = static_cast<size_t>(mid - lo) * n;
    MPI_Sendrecv(&acc[send_lo * n], static_cast<int>(cnt), MPI_FLOAT, partner, d, tmp.data(), static_cast<int>(cnt),
                 MPI_FLOAT, partner, d, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    add_into(&acc[keep_lo * n], tmp.data(), cnt);
    hi = keep_lo + (mid - lo);  // the kept half
    lo = keep_lo;
  }
  std::memcpy(out, &acc[rank_ * n], n * sizeof(float));
}

static void allreduce(const float *in, float *out, size_t n, std::vector<float> &acc) {
  std::memcpy(acc.data(), in, n * size_ * sizeof(float));
  MPI_Allreduce(MPI_IN_PLACE, acc.data(), static_cast<int>(n * size_), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  std::memcpy(out, &acc[rank_ * n], n * sizeof(float));
}

// in: n floats; out: size_ blocks of n, block r from rank r.
static void ag_staged(const float *in, float *out, size_t n, std::vector<float> &si, std::vector<float> &so) {
  const size_t chunk = (8u << 20) / (sizeof(float) * size_);
  for (size_t off = 0; off < n; off += chunk) {
    const size_t cnt = std::min(chunk, n - off);
    std::memcpy(si.data(), in + off, cnt * sizeof(float));
    MPI_Allgather(si.data(), static_cast<int>(cnt), MPI_FLOAT, so.data(), static_cast<int>(cnt), MPI_FLOAT,
                  MPI_COMM_WORLD);
    for (int r = 0; r < size_; ++r) std::memcpy(out + r * n + off, &so[r * cnt], cnt * sizeof(float));
  }
}

static void ag_ring(const float *in, float *out, size_t n) {
  std::memcpy(out + rank_ * n, in, n * sizeof(float));
  const int next = (rank_ + 1) % size_, prev = (rank_ - 1 + size_) % size_;
  for (int k = 0; k < size_ - 1; ++k) {
    const int s = ((rank_ - k) % size_ + size_) % size_, r = ((rank_ - k - 1) % size_ + size_) % size_;
    MPI_Sendrecv(out + s * n, static_cast<int>(n), MPI_FLOAT, next, k, out + r * n, static_cast<int>(n), MPI_FLOAT,
                 prev, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

int main(int argc, char **argv) {
  const double mem0 = mem_avail_gb();
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  if (argc < 4) {
    if (rank_ == 0) std::fprintf(stderr, "usage: rs_bench variant reps floats...\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  const std::string v = argv[1];
  const int reps = std::atoi(argv[2]);
  if (v == "halving" && (size_ & (size_ - 1))) {
    if (rank_ == 0) std::printf("{\"variant\":\"halving\",\"skipped\":\"ranks not a power of two\"}\n");
    MPI_Finalize();
    return 0;
  }
  for (int a = 3; a < argc; ++a) {
    const size_t total = std::strtoull(argv[a], nullptr, 10), n = total / size_;
    const bool ag = v.rfind("ag", 0) == 0;
    std::vector<float> in(ag ? n : n * size_), out(ag ? n * size_ : n), w1, w2;
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>((rank_ % 7) + (i % 5));
    if (v == "ag_staged") w1.resize((8u << 20) / sizeof(float)), w2.resize((8u << 20) / sizeof(float));
    MPI_Request preq = MPI_REQUEST_NULL;
    if (v == "ag_persist")
      MPIX_Allgather_init(in.data(), static_cast<int>(n), MPI_FLOAT, out.data(), static_cast<int>(n), MPI_FLOAT,
                          MPI_COMM_WORLD, MPI_INFO_NULL, &preq);
    if (v == "a2a_staged") w1.resize((8u << 20) / sizeof(float)), w2.resize((8u << 20) / sizeof(float));
    if (v == "a2a") w1.resize(n * size_);
    if (v == "ring" || v == "halving") w1.resize(n * size_), w2.resize(n * size_);
    if (v == "allreduce") w1.resize(n * size_);
    auto run = [&] {
      if (v == "a2a_staged") a2a_staged(in.data(), out.data(), n, w1, w2);
      else if (v == "a2a") a2a(in.data(), out.data(), n, w1);
      else if (v == "rsb") MPI_Reduce_scatter_block(in.data(), out.data(), static_cast<int>(n), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      else if (v == "ring") ring(in.data(), out.data(), n, w1, w2);
      else if (v == "halving") halving(in.data(), out.data(), n, w1, w2);
      else if (v == "allreduce") allreduce(in.data(), out.data(), n, w1);
      else if (v == "ag_staged") ag_staged(in.data(), out.data(), n, w1, w2);
      else if (v == "ag") MPI_Allgather(in.data(), static_cast<int>(n), MPI_FLOAT, out.data(), static_cast<int>(n), MPI_FLOAT, MPI_COMM_WORLD);
      else if (v == "ag_ring") ag_ring(in.data(), out.data(), n);
      else if (v == "ag_persist") MPI_Start(&preq), MPI_Wait(&preq, MPI_STATUS_IGNORE);
      else MPI_Abort(MPI_COMM_WORLD, 3);
    };
    std::vector<double> ts;
    for (int r = 0; r <= reps; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      const double t0 = MPI_Wtime();
      run();
      double t = MPI_Wtime() - t0, tmax;
      MPI_Allreduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (r) ts.push_back(tmax);  // first call is warm-up (connections, registration)
    }
    std::sort(ts.begin(), ts.end());
    long sr = 0;
    for (int r = 0; r < size_; ++r) sr += r % 7;
    int ok = 1;
    for (size_t j = 0; !ag && j < n; ++j)
      if (out[j] != static_cast<float>(sr + size_ * ((rank_ * n + j) % 5))) { ok = 0; break; }
    for (size_t j = 0; ag && j < n * size_; ++j)
      if (out[j] != static_cast<float>((j / n) % 7 + (j % n) % 5)) { ok = 0; break; }
    if (preq != MPI_REQUEST_NULL) MPI_Request_free(&preq);
    int all_ok;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    double mem1 = mem_avail_gb(), m0, m1;
    MPI_Allreduce(&mem0, &m0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mem1, &m1, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    if (rank_ == 0)
      std::printf("{\"variant\":\"%s\",\"ranks\":%d,\"threads\":%d,\"floats\":%zu,\"bytes_in_per_rank\":%zu,"
                  "\"time_s\":%.6f,\"gbps_in\":%.2f,\"ok\":%d,\"mem_avail_before_gb\":%.2f,\"mem_avail_after_gb\":%.2f}\n",
                  v.c_str(), size_, omp_get_max_threads(), total, total * sizeof(float), ts[ts.size() / 2],
                  total * sizeof(float) / ts[ts.size() / 2] / 1e9, all_ok, m0, m1);
    std::fflush(stdout);
  }
  MPI_Finalize();
  return 0;
}
