// torch.distributed backend "fjmpi_async": fjmpi (h2h-profile/port/c10d_mpi.cpp, the
// container's pinned backend) plus allgathers that overlap with computation.
//
// Everything except _allgather_base / allgather_into_tensor_coalesced is unchanged: synchronous,
// each call returns a completed Work, data through two reused staging buffers
// (TORCH_FUGAKU_STAGE_MB each, default 8; 0 sends the tensors themselves).
//
// With TORCH_FUGAKU_ASYNC=1 (the default) those allgathers are queued and return a pending Work.
// The queue is driven through one persistent MPIX_Allgather_init request on a dedicated
// communicator, which Fujitsu MPI offloads to the Tofu interconnect from its second MPI_Start
// (MPI manual 5.5.3): the chunk in flight moves while the caller computes. Plain
// MPI_Iallgather does not move until MPI_Wait, and the assistant-core progress thread does not
// help (h2h-overlap, job 53462090). Only one offloaded request may exist per process, so every
// gather goes through one fixed pair of buffers (TORCH_FUGAKU_AG_STAGE_MB for all ranks' data,
// default 64) in chunks of that / group size per rank. The next chunk starts whenever the
// backend is entered: any collective call, Work::isCompleted or Work::wait.
//
// Every other collective first drains the queue. Without that, a rank blocked in a collective
// on the main communicator could hold back a gather chunk another rank is waiting for.
// TORCH_FUGAKU_ASYNC=0 gives fjmpi's synchronous allgather, for A/B runs on the same binary.
// stats() reports time blocked in the backend, by kind, for both. With TORCH_FUGAKU_TRACE=1,
// trace() returns (and clears) timestamped events -- gather issue, transfer start, transfer seen
// complete, MPI_Test misses, waits, drains, other collectives -- on CLOCK_MONOTONIC, the clock of
// Python's time.perf_counter, so they line up with the training script's phases.
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#else
extern "C" int MPIX_Allgather_init(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
                                   MPI_Request *request);
#endif
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/extension.h>

#include <time.h>

#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>

namespace {

size_t kStage = 8 << 20;            // bytes per staging buffer; 0: no staging
size_t kAgStage = 64 << 20;         // bytes of the async gather's receive buffer (all ranks)
bool kAsync = true;
constexpr size_t kMaxCount = 1 << 30;  // MPI counts are int: direct calls go in pieces
// MPI_THREAD_SERIALIZED: one caller at a time. Recursive because completing a Work's future
// runs its callbacks inline, and a callback may issue another collective.
std::recursive_mutex mpi_mutex;
using Lock = std::lock_guard<std::recursive_mutex>;
char *stage_in = nullptr, *stage_out = nullptr;

// Seconds blocked inside the backend, and what moved; summed over all groups of the process.
struct Stats {
  double wait_s = 0;        // Work::wait on queued gathers
  double drain_s = 0;       // finishing queued gathers before another collective
  double gather_sync_s = 0; // synchronous allgathers (TORCH_FUGAKU_ASYNC=0, or the list form)
  double reduce_scatter_s = 0, allreduce_s = 0, other_s = 0;
  int64_t gathers_async = 0, gathers_sync = 0, chunks = 0, waits = 0, waits_already_done = 0;
  int64_t gather_bytes = 0;  // output bytes
} stats;

// Trace event kinds; id is the gather's sequence number, bytes its per-rank size (or, for
// kColl*, the collective: 1 reduce-scatter, 2 allreduce, 3 other, 4 synchronous allgather).
enum Kind { kIssue, kStart, kDone, kTestMiss, kWaitBegin, kWaitEnd, kDrainBegin, kDrainEnd, kCollBegin, kCollEnd };
struct Event {
  double t;
  int kind;
  int64_t id, bytes;
};
bool kTrace = false;
std::vector<Event> events;

double now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

void trace(int kind, int64_t id, int64_t bytes) {
  if (kTrace && events.size() < (1u << 22)) events.push_back(Event{now(), kind, id, bytes});
}

// Accumulates the time of a scope into a Stats field and traces its two ends.
struct Timed {
  double &acc, t0;
  int end_kind;
  int64_t id, code;
  Timed(double &a, int begin_kind = kCollBegin, int64_t id_ = -1, int64_t code_ = 3)
      : acc(a), t0(MPI_Wtime()), end_kind(begin_kind + 1), id(id_), code(code_) {
    trace(begin_kind, id, code);
  }
  ~Timed() {
    acc += MPI_Wtime() - t0;
    trace(end_kind, id, code);
  }
};

void check(int rc, const char *what) {
  if (rc != MPI_SUCCESS) {
    char msg[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(rc, msg, &len);
    TORCH_CHECK(false, "fjmpi: ", what, " failed: ", std::string(msg, len));
  }
}

void ensure_init() {
  int ok = 0, provided = 0;
  MPI_Initialized(&ok);
  if (!ok) check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided), "MPI_Init_thread");
  if (const char *mb = std::getenv("TORCH_FUGAKU_STAGE_MB")) kStage = std::strtoull(mb, nullptr, 10) << 20;
  if (const char *mb = std::getenv("TORCH_FUGAKU_AG_STAGE_MB")) kAgStage = std::strtoull(mb, nullptr, 10) << 20;
  if (const char *a = std::getenv("TORCH_FUGAKU_ASYNC")) kAsync = std::strcmp(a, "0") != 0;
  if (const char *a = std::getenv("TORCH_FUGAKU_TRACE")) kTrace = std::strcmp(a, "0") != 0;
  if (!stage_in && kStage > 0) {
    TORCH_CHECK(posix_memalign(reinterpret_cast<void **>(&stage_in), 1 << 21, kStage) == 0 &&
                    posix_memalign(reinterpret_cast<void **>(&stage_out), 1 << 21, kStage) == 0,
                "fjmpi: cannot allocate staging buffers");
  }
}

MPI_Datatype mpi_type(const at::Tensor &t) {
  switch (t.scalar_type()) {
    case at::kFloat: return MPI_FLOAT;
    case at::kDouble: return MPI_DOUBLE;
    case at::kInt: return MPI_INT32_T;
    case at::kLong: return MPI_INT64_T;
    case at::kShort: return MPI_INT16_T;
    case at::kChar: return MPI_INT8_T;
    case at::kByte: return MPI_UINT8_T;
    case at::kBool: return MPI_C_BOOL;
    default: TORCH_CHECK(false, "fjmpi: reductions on ", t.scalar_type(), " are not supported");
  }
}

MPI_Op mpi_op(const c10d::ReduceOp &op, const at::Tensor &t) {
  switch (op.op_) {
    case c10d::ReduceOp::SUM:
    case c10d::ReduceOp::AVG: return t.scalar_type() == at::kBool ? MPI_LOR : MPI_SUM;
    case c10d::ReduceOp::PRODUCT: return t.scalar_type() == at::kBool ? MPI_LAND : MPI_PROD;
    case c10d::ReduceOp::MIN: return t.scalar_type() == at::kBool ? MPI_LAND : MPI_MIN;
    case c10d::ReduceOp::MAX: return t.scalar_type() == at::kBool ? MPI_LOR : MPI_MAX;
    case c10d::ReduceOp::BAND: return MPI_BAND;
    case c10d::ReduceOp::BOR: return MPI_BOR;
    case c10d::ReduceOp::BXOR: return MPI_BXOR;
    default: TORCH_CHECK(false, "fjmpi: unsupported reduce op");
  }
}

bool half_like(const at::Tensor &t) {
  return t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16;
}

void check_tensor(const at::Tensor &t) {
  TORCH_CHECK(t.device().is_cpu(), "fjmpi: tensors must be on the CPU");
  TORCH_CHECK(t.is_contiguous(), "fjmpi: tensors must be contiguous");
}

class DoneWork : public c10d::Work {
 public:
  explicit DoneWork(std::vector<at::Tensor> out)
      : out_(std::move(out)),
        fut_(c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()))) {
    fut_->markCompleted(c10::IValue(out_));
  }
  bool isCompleted() override { return true; }
  bool isSuccess() const override { return true; }
  bool wait(std::chrono::milliseconds) override { return true; }
  std::vector<at::Tensor> result() override { return out_; }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return fut_; }

 private:
  std::vector<at::Tensor> out_;
  c10::intrusive_ptr<c10::ivalue::Future> fut_;
};

c10::intrusive_ptr<c10d::Work> done(std::vector<at::Tensor> out) {
  return c10::make_intrusive<DoneWork>(std::move(out));
}

class FjMpiBackend;

// A queued allgather; completes when its last chunk has been copied out.
class GatherWork : public c10d::Work {
 public:
  GatherWork(c10::intrusive_ptr<FjMpiBackend> backend, std::vector<at::Tensor> out)
      : backend_(std::move(backend)),
        out_(std::move(out)),
        fut_(c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()))) {}
  bool isCompleted() override;
  bool isSuccess() const override { return done_; }
  bool wait(std::chrono::milliseconds) override;
  std::vector<at::Tensor> result() override { return out_; }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return fut_; }

  int64_t id_ = -1;
  void finish() {
    done_ = true;
    fut_->markCompleted(c10::IValue(out_));
  }
  bool done_ = false;

 private:
  c10::intrusive_ptr<FjMpiBackend> backend_;
  std::vector<at::Tensor> out_;
  c10::intrusive_ptr<c10::ivalue::Future> fut_;
};

class FjMpiBackend : public c10d::Backend {
 public:
  // global_ranks: the group's members in MPI_COMM_WORLD order; empty means the world.
  FjMpiBackend(int rank, int size, const std::vector<int64_t> &global_ranks, const std::string &group_id)
      : c10d::Backend(rank, size) {
    Lock lock(mpi_mutex);
    ensure_init();
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (global_ranks.empty() || static_cast<int>(global_ranks.size()) == world) {
      check(MPI_Comm_dup(MPI_COMM_WORLD, &comm_), "MPI_Comm_dup");
    } else {
      // Only members create the group, so the tag comes from the group's name, not a counter.
      std::vector<int> ranks(global_ranks.begin(), global_ranks.end());
      MPI_Group world_group, group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);
      MPI_Group_incl(world_group, static_cast<int>(ranks.size()), ranks.data(), &group);
      int tag = static_cast<int>(std::hash<std::string>{}(group_id) & 0x7fff);
      check(MPI_Comm_create_group(MPI_COMM_WORLD, group, tag, &comm_), "MPI_Comm_create_group");
      MPI_Group_free(&group);
      MPI_Group_free(&world_group);
    }
    int r = -1, s = 0;
    MPI_Comm_rank(comm_, &r);
    MPI_Comm_size(comm_, &s);
    TORCH_CHECK(r == rank && s == size, "fjmpi: MPI rank/size ", r, "/", s, " != process group rank/size ",
                rank, "/", size, "; launch with mpiexec and take RANK from it (pf-run does)");
  }

  ~FjMpiBackend() override {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) return;
    if (ag_req_ != MPI_REQUEST_NULL) {
      if (in_flight_) MPI_Wait(&ag_req_, MPI_STATUS_IGNORE);
      MPI_Request_free(&ag_req_);
    }
    if (ag_comm_ != MPI_COMM_NULL) MPI_Comm_free(&ag_comm_);
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
    std::free(ag_in_);
    std::free(ag_out_);
  }

  const std::string getBackendName() const override { return "fjmpi_async"; }

  // Advance queued gathers: finish the chunk in flight if it is done (or, with block, wait for
  // it) and start the next. With block, returns once until (or the whole queue, if null) is done.
  void pump(bool block, const GatherWork *until = nullptr) {
    Lock lock(mpi_mutex);
    for (;;) {
      if (in_flight_) {
        int flag = 1;
        if (block)
          check(MPI_Wait(&ag_req_, MPI_STATUS_IGNORE), "MPI_Wait");
        else
          check(MPI_Test(&ag_req_, &flag, MPI_STATUS_IGNORE), "MPI_Test");
        if (!flag) {
          trace(kTestMiss, queue_.front().id, 0);
          return;
        }
        in_flight_ = false;
        Gather &g = queue_.front();
        trace(kDone, g.id, static_cast<int64_t>(flight_len_));
        const auto bytes = at::TensorOptions().dtype(at::kByte);
        at::from_blob(g.out.data_ptr(), {size_, static_cast<int64_t>(g.n)}, bytes)
            .narrow(1, static_cast<int64_t>(g.off), static_cast<int64_t>(flight_len_))
            .copy_(at::from_blob(ag_out_, {size_, static_cast<int64_t>(ag_chunk_)}, bytes)
                       .narrow(1, 0, static_cast<int64_t>(flight_len_)));
        g.off += flight_len_;
        if (g.off == g.n) {
          auto work = g.work;
          const bool last = g.last;
          queue_.pop_front();
          if (last) work->finish();
        }
      }
      if (!queue_.empty()) start_next();
      if (!block || !in_flight_ || (until && until->done_)) return;
    }
  }

  // Every collective other than the queued gathers runs only on an empty queue.
  void drain() {
    if (!in_flight_ && queue_.empty()) return;
    Timed timed(stats.drain_s, kDrainBegin);
    pump(true);
  }

  c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor> &tensors,
                                           const c10d::AllreduceOptions &opts) override {
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.allreduce_s, kCollBegin, -1, 2);
    for (auto &t : tensors) reduce_all(t, opts.reduceOp);
    return done(tensors);
  }

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(std::vector<at::Tensor> &tensors,
                                                     const c10d::AllreduceCoalescedOptions &opts) override {
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.allreduce_s, kCollBegin, -1, 2);
    for (auto &t : tensors) reduce_all(t, opts.reduceOp);
    return done(tensors);
  }

  c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor> &tensors,
                                           const c10d::BroadcastOptions &opts) override {
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.other_s);
    const int root = static_cast<int>(opts.rootRank);
    for (auto &t : tensors) {
      check_tensor(t);
      char *p = static_cast<char *>(t.data_ptr());
      const size_t n = t.nbytes();
      if (kStage == 0) {
        for (size_t off = 0; off < n; off += kMaxCount)
          check(MPI_Bcast(p + off, static_cast<int>(std::min(kMaxCount, n - off)), MPI_BYTE, root, comm_), "MPI_Bcast");
        continue;
      }
      for (size_t off = 0; off < n; off += kStage) {
        const size_t len = std::min(kStage, n - off);
        if (rank_ == root) std::memcpy(stage_in, p + off, len);
        check(MPI_Bcast(stage_in, static_cast<int>(len), MPI_BYTE, root, comm_), "MPI_Bcast");
        if (rank_ != root) std::memcpy(p + off, stage_in, len);
      }
    }
    return done(tensors);
  }

  c10::intrusive_ptr<c10d::Work> reduce(std::vector<at::Tensor> &tensors,
                                        const c10d::ReduceOptions &opts) override {
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.other_s);
    for (auto &t : tensors) reduce_to(t, opts.reduceOp, static_cast<int>(opts.rootRank));
    return done(tensors);
  }

  c10::intrusive_ptr<c10d::Work> _allgather_base(at::Tensor &out, at::Tensor &in,
                                                 const c10d::AllgatherOptions &) override {
    std::vector<at::Tensor> outs{out}, ins{in};
    return gather_queued_or_sync(outs, ins);
  }

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(std::vector<at::Tensor> &outs,
                                                                 std::vector<at::Tensor> &ins,
                                                                 const c10d::AllgatherOptions &) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: allgather_into_tensor_coalesced size mismatch");
    return gather_queued_or_sync(outs, ins);
  }

  c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>> &outs,
                                           std::vector<at::Tensor> &ins,
                                           const c10d::AllgatherOptions &) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: allgather size mismatch");
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.gather_sync_s, kCollBegin, -1, 4);
    std::vector<at::Tensor> flat_outs;
    for (size_t i = 0; i < ins.size(); ++i) {
      TORCH_CHECK(static_cast<int>(outs[i].size()) == size_, "fjmpi: allgather needs one output per rank");
      at::Tensor in = ins[i].contiguous();
      at::Tensor flat = at::empty({size_ * in.numel()}, in.options());
      gather_all(flat, in);
      for (int r = 0; r < size_; ++r) outs[i][r].copy_(flat.narrow(0, r * in.numel(), in.numel()).view_as(outs[i][r]));
      flat_outs.push_back(flat);
    }
    return done(flat_outs);
  }

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(at::Tensor &out, at::Tensor &in,
                                                      const c10d::ReduceScatterOptions &opts) override {
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.reduce_scatter_s, kCollBegin, -1, 1);
    scatter_reduce(out, in, opts.reduceOp);
    return done({out});
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(std::vector<at::Tensor> &outs,
                                                                 std::vector<at::Tensor> &ins,
                                                                 const c10d::ReduceScatterOptions &opts) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: reduce_scatter_tensor_coalesced size mismatch");
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.reduce_scatter_s, kCollBegin, -1, 1);
    for (size_t i = 0; i < ins.size(); ++i) scatter_reduce(outs[i], ins[i], opts.reduceOp);
    return done(outs);
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter(std::vector<at::Tensor> &outs,
                                                std::vector<std::vector<at::Tensor>> &ins,
                                                const c10d::ReduceScatterOptions &opts) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: reduce_scatter size mismatch");
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.reduce_scatter_s, kCollBegin, -1, 1);
    for (size_t i = 0; i < outs.size(); ++i) {
      TORCH_CHECK(static_cast<int>(ins[i].size()) == size_, "fjmpi: reduce_scatter needs one input per rank");
      std::vector<at::Tensor> parts;
      for (auto &x : ins[i]) parts.push_back(x.reshape({-1}));
      at::Tensor flat = at::cat(parts);
      at::Tensor out = outs[i].is_contiguous() ? outs[i] : outs[i].contiguous();
      scatter_reduce(out, flat, opts.reduceOp);
      if (!outs[i].is_same(out)) outs[i].copy_(out);
    }
    return done(outs);
  }

  c10::intrusive_ptr<c10d::Work> alltoall_base(at::Tensor &out, at::Tensor &in,
                                               std::vector<int64_t> &out_splits,
                                               std::vector<int64_t> &in_splits,
                                               const c10d::AllToAllOptions &) override {
    check_tensor(out);
    check_tensor(in);
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.other_s);
    if (out_splits.empty() && in_splits.empty()) {
      TORCH_CHECK(in.nbytes() == out.nbytes() && in.nbytes() % size_ == 0,
                  "fjmpi: alltoall_base needs equal input and output divisible by the group size");
      const size_t n = in.nbytes() / size_;
      if (kStage == 0) {
        TORCH_CHECK(n <= kMaxCount, "fjmpi: alltoall block too large for one MPI call");
        check(MPI_Alltoall(in.data_ptr(), static_cast<int>(n), MPI_BYTE, out.data_ptr(), static_cast<int>(n), MPI_BYTE,
                           comm_), "MPI_Alltoall");
        return done({out});
      }
      const size_t chunk = kStage / size_;
      TORCH_CHECK(chunk > 0, "fjmpi: group too large for the staging buffer");
      const char *src = static_cast<const char *>(in.data_ptr());
      char *dst = static_cast<char *>(out.data_ptr());
      for (size_t off = 0; off < n; off += chunk) {
        const size_t len = std::min(chunk, n - off);
        for (int r = 0; r < size_; ++r) std::memcpy(stage_in + r * len, src + r * n + off, len);
        check(MPI_Alltoall(stage_in, static_cast<int>(len), MPI_BYTE, stage_out, static_cast<int>(len),
                           MPI_BYTE, comm_), "MPI_Alltoall");
        for (int r = 0; r < size_; ++r) std::memcpy(dst + r * n + off, stage_out + r * len, len);
      }
    } else {
      // Uneven splits along dim 0: unstaged MPI_Alltoallv on the tensors themselves.
      const size_t in_row = in.size(0) ? in.nbytes() / in.size(0) : 0;
      const size_t out_row = out.size(0) ? out.nbytes() / out.size(0) : 0;
      auto counts = [&](const std::vector<int64_t> &splits, int64_t rows, size_t row) {
        std::vector<int> c(size_), d(size_);
        for (int r = 0; r < size_; ++r) c[r] = static_cast<int>((splits.empty() ? rows / size_ : splits[r]) * row);
        for (int r = 1; r < size_; ++r) d[r] = d[r - 1] + c[r - 1];
        return std::make_pair(c, d);
      };
      auto [sc, sd] = counts(in_splits, in.size(0), in_row);
      auto [rc, rd] = counts(out_splits, out.size(0), out_row);
      check(MPI_Alltoallv(in.data_ptr(), sc.data(), sd.data(), MPI_BYTE, out.data_ptr(), rc.data(), rd.data(),
                          MPI_BYTE, comm_), "MPI_Alltoallv");
    }
    return done({out});
  }

  c10::intrusive_ptr<c10d::Work> barrier(const c10d::BarrierOptions &) override {
    Lock lock(mpi_mutex);
    drain();
    Timed timed(stats.other_s);
    check(MPI_Barrier(comm_), "MPI_Barrier");
    return done({});
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;

  // One gather in the queue: out = [in from rank 0, in from rank 1, ...], byte-wise; off is
  // how much of each rank's block has arrived. A coalesced call is several, last finishing it.
  struct Gather {
    at::Tensor out, in;
    size_t n, off;
    int64_t id;
    c10::intrusive_ptr<GatherWork> work;
    bool last;
  };
  std::deque<Gather> queue_;
  MPI_Comm ag_comm_ = MPI_COMM_NULL;   // gathers only, so they never interleave with comm_
  MPI_Request ag_req_ = MPI_REQUEST_NULL;
  char *ag_in_ = nullptr, *ag_out_ = nullptr;
  size_t ag_chunk_ = 0;                // bytes per rank per MPI_Start
  bool in_flight_ = false;
  size_t flight_len_ = 0;
  int64_t next_id_ = 0;

  c10::intrusive_ptr<c10d::Work> gather_queued_or_sync(std::vector<at::Tensor> &outs, std::vector<at::Tensor> &ins) {
    Lock lock(mpi_mutex);
    for (size_t i = 0; i < ins.size(); ++i) {
      check_tensor(outs[i]);
      check_tensor(ins[i]);
      TORCH_CHECK(outs[i].nbytes() == ins[i].nbytes() * size_, "fjmpi: allgather output must be group size x input");
      stats.gather_bytes += static_cast<int64_t>(outs[i].nbytes());
    }
    if (!kAsync) {
      Timed timed(stats.gather_sync_s, kCollBegin, -1, 4);
      stats.gathers_sync += static_cast<int64_t>(ins.size());
      for (size_t i = 0; i < ins.size(); ++i) gather_all(outs[i], ins[i]);
      return done(outs);
    }
    if (ag_req_ == MPI_REQUEST_NULL) open_gather_channel();
    pump(false);
    auto work = c10::make_intrusive<GatherWork>(c10::intrusive_ptr<FjMpiBackend>::reclaim_copy(this), outs);
    bool queued = false;
    for (size_t i = 0; i < ins.size(); ++i) {
      if (ins[i].nbytes() == 0) continue;
      queue_.push_back(Gather{outs[i], ins[i], ins[i].nbytes(), 0, next_id_, work, false});
      trace(kIssue, next_id_, static_cast<int64_t>(ins[i].nbytes()));
      work->id_ = next_id_++;
      queued = true;
    }
    stats.gathers_async += static_cast<int64_t>(ins.size());
    if (!queued) {
      work->finish();
      return work;
    }
    queue_.back().last = true;
    if (!in_flight_) start_next();
    return work;
  }

  void open_gather_channel() {
    check(MPI_Comm_dup(comm_, &ag_comm_), "MPI_Comm_dup");
    ag_chunk_ = (kAgStage / size_) & ~static_cast<size_t>(255);
    TORCH_CHECK(ag_chunk_ > 0 && ag_chunk_ <= kMaxCount, "fjmpi: TORCH_FUGAKU_AG_STAGE_MB unusable for group size ", size_);
    TORCH_CHECK(posix_memalign(reinterpret_cast<void **>(&ag_in_), 1 << 21, ag_chunk_) == 0 &&
                    posix_memalign(reinterpret_cast<void **>(&ag_out_), 1 << 21, ag_chunk_ * size_) == 0,
                "fjmpi: cannot allocate gather buffers");
    check(MPIX_Allgather_init(ag_in_, static_cast<int>(ag_chunk_), MPI_BYTE, ag_out_, static_cast<int>(ag_chunk_),
                              MPI_BYTE, ag_comm_, MPI_INFO_NULL, &ag_req_),
          "MPIX_Allgather_init");
  }

  // Every rank sends a full ag_chunk_; only the first flight_len_ bytes of each block are used.
  void start_next() {
    Gather &g = queue_.front();
    flight_len_ = std::min(ag_chunk_, g.n - g.off);
    std::memcpy(ag_in_, static_cast<const char *>(g.in.data_ptr()) + g.off, flight_len_);
    check(MPI_Start(&ag_req_), "MPI_Start");
    trace(kStart, g.id, static_cast<int64_t>(flight_len_));
    in_flight_ = true;
    ++stats.chunks;
  }

  void reduce_all(at::Tensor &t, const c10d::ReduceOp &op) {
    check_tensor(t);
    if (half_like(t)) {  // no MPI type: reduce in float32
      at::Tensor f = t.to(at::kFloat);
      reduce_all(f, op);
      t.copy_(f);
      return;
    }
    {
      Lock lock(mpi_mutex);
      const MPI_Datatype type = mpi_type(t);
      const MPI_Op mop = mpi_op(op, t);
      const size_t es = t.element_size(), chunk = kStage ? kStage / es : kMaxCount;
      char *p = static_cast<char *>(t.data_ptr());
      const size_t n = t.numel();
      for (size_t off = 0; off < n; off += chunk) {
        const size_t cnt = std::min(chunk, n - off);
        if (kStage == 0) {
          check(MPI_Allreduce(MPI_IN_PLACE, p + off * es, static_cast<int>(cnt), type, mop, comm_), "MPI_Allreduce");
          continue;
        }
        std::memcpy(stage_in, p + off * es, cnt * es);
        check(MPI_Allreduce(MPI_IN_PLACE, stage_in, static_cast<int>(cnt), type, mop, comm_), "MPI_Allreduce");
        std::memcpy(p + off * es, stage_in, cnt * es);
      }
    }
    if (op.op_ == c10d::ReduceOp::AVG) t.div_(size_);
  }

  void reduce_to(at::Tensor &t, const c10d::ReduceOp &op, int root) {
    check_tensor(t);
    if (half_like(t)) {
      at::Tensor f = t.to(at::kFloat);
      reduce_to(f, op, root);
      if (rank_ == root) t.copy_(f);
      return;
    }
    {
      Lock lock(mpi_mutex);
      const MPI_Datatype type = mpi_type(t);
      const MPI_Op mop = mpi_op(op, t);
      const size_t es = t.element_size(), chunk = kStage ? kStage / es : kMaxCount;
      char *p = static_cast<char *>(t.data_ptr());
      const size_t n = t.numel();
      for (size_t off = 0; off < n; off += chunk) {
        const size_t cnt = std::min(chunk, n - off);
        if (kStage == 0) {
          void *sbuf = rank_ == root ? MPI_IN_PLACE : p + off * es;
          check(MPI_Reduce(sbuf, p + off * es, static_cast<int>(cnt), type, mop, root, comm_), "MPI_Reduce");
          continue;
        }
        std::memcpy(stage_in, p + off * es, cnt * es);
        check(MPI_Reduce(stage_in, stage_out, static_cast<int>(cnt), type, mop, root, comm_), "MPI_Reduce");
        if (rank_ == root) std::memcpy(p + off * es, stage_out, cnt * es);
      }
    }
    if (rank_ == root && op.op_ == c10d::ReduceOp::AVG) t.div_(size_);
  }

  // out = [in from rank 0, in from rank 1, ...], byte-wise.
  void gather_all(at::Tensor &out, at::Tensor &in) {
    check_tensor(out);
    check_tensor(in);
    TORCH_CHECK(out.nbytes() == in.nbytes() * size_, "fjmpi: allgather output must be group size x input");
    Lock lock(mpi_mutex);
    const size_t n = in.nbytes();
    const char *src = static_cast<const char *>(in.data_ptr());
    char *dst = static_cast<char *>(out.data_ptr());
    if (kStage == 0) {
      TORCH_CHECK(n <= kMaxCount, "fjmpi: allgather block too large for one MPI call");
      check(MPI_Allgather(src, static_cast<int>(n), MPI_BYTE, dst, static_cast<int>(n), MPI_BYTE, comm_),
            "MPI_Allgather");
      return;
    }
    const size_t chunk = kStage / size_;
    TORCH_CHECK(chunk > 0, "fjmpi: group too large for the staging buffer");
    for (size_t off = 0; off < n; off += chunk) {
      const size_t len = std::min(chunk, n - off);
      std::memcpy(stage_in, src + off, len);
      check(MPI_Allgather(stage_in, static_cast<int>(len), MPI_BYTE, stage_out, static_cast<int>(len), MPI_BYTE,
                          comm_), "MPI_Allgather");
      for (int r = 0; r < size_; ++r) std::memcpy(dst + r * n + off, stage_out + r * len, len);
    }
  }

  // out = reduction over ranks of their in[rank * out.numel() : (rank + 1) * out.numel()].
  // Arithmetic ops go as an all-to-all and a local reduction over ranks in rank order: Fujitsu
  // MPI's MPI_Reduce_scatter_block moves data ~10x slower than its MPI_Allgather at 64 ranks.
  void scatter_reduce(at::Tensor &out, at::Tensor &in, const c10d::ReduceOp &op) {
    check_tensor(out);
    check_tensor(in);
    TORCH_CHECK(in.numel() == out.numel() * size_, "fjmpi: reduce_scatter input must be group size x output");
    if (half_like(out)) {
      at::Tensor fo = out.to(at::kFloat), fi = in.to(at::kFloat);
      scatter_reduce(fo, fi, op);
      out.copy_(fo);
      return;
    }
    const auto kind = op.op_;
    const bool local = out.scalar_type() != at::kBool &&
                       (kind == c10d::ReduceOp::SUM || kind == c10d::ReduceOp::AVG || kind == c10d::ReduceOp::MAX ||
                        kind == c10d::ReduceOp::MIN || kind == c10d::ReduceOp::PRODUCT);
    auto reduce_blocks = [&](at::Tensor blocks, at::Tensor o) {  // blocks: [size, cnt] -> o: [cnt]
      switch (kind) {
        case c10d::ReduceOp::MAX: at::amax_out(o, blocks, {0}); break;
        case c10d::ReduceOp::MIN: at::amin_out(o, blocks, {0}); break;
        case c10d::ReduceOp::PRODUCT: at::prod_out(o, blocks, 0); break;
        default: at::sum_out(o, blocks, {0}); break;
      }
    };
    at::Tensor flat_out = out.view({-1});
    {
      Lock lock(mpi_mutex);
      const MPI_Datatype type = mpi_type(out);
      const MPI_Op mop = mpi_op(op, out);
      const size_t es = out.element_size(), n = out.numel();
      const char *src = static_cast<const char *>(in.data_ptr());
      char *dst = static_cast<char *>(out.data_ptr());
      if (kStage == 0) {
        TORCH_CHECK(n * es <= kMaxCount, "fjmpi: reduce_scatter block too large for one MPI call");
        if (local) {
          at::Tensor recv = at::empty({size_, static_cast<int64_t>(n)}, out.options());
          check(MPI_Alltoall(src, static_cast<int>(n * es), MPI_BYTE, recv.data_ptr(), static_cast<int>(n * es),
                             MPI_BYTE, comm_), "MPI_Alltoall");
          reduce_blocks(recv, flat_out);
        } else {
          check(MPI_Reduce_scatter_block(src, dst, static_cast<int>(n), type, mop, comm_), "MPI_Reduce_scatter_block");
        }
      }
      const size_t chunk = kStage ? kStage / (es * size_) : 0;
      TORCH_CHECK(kStage == 0 || chunk > 0, "fjmpi: group too large for the staging buffer");
      for (size_t off = 0; kStage && off < n; off += chunk) {
        const size_t cnt = std::min(chunk, n - off);
        for (int r = 0; r < size_; ++r) std::memcpy(stage_in + r * cnt * es, src + (r * n + off) * es, cnt * es);
        if (local) {
          check(MPI_Alltoall(stage_in, static_cast<int>(cnt * es), MPI_BYTE, stage_out, static_cast<int>(cnt * es),
                             MPI_BYTE, comm_), "MPI_Alltoall");
          reduce_blocks(at::from_blob(stage_out, {size_, static_cast<int64_t>(cnt)}, out.options()),
                        flat_out.narrow(0, static_cast<int64_t>(off), static_cast<int64_t>(cnt)));
        } else {
          check(MPI_Reduce_scatter_block(stage_in, stage_out, static_cast<int>(cnt), type, mop, comm_),
                "MPI_Reduce_scatter_block");
          std::memcpy(dst + off * es, stage_out, cnt * es);
        }
      }
    }
    if (kind == c10d::ReduceOp::AVG) out.div_(size_);
  }
};

bool GatherWork::isCompleted() {
  if (!done_) backend_->pump(false);
  return done_;
}

bool GatherWork::wait(std::chrono::milliseconds) {
  Lock lock(mpi_mutex);
  ++stats.waits;
  if (done_) {
    ++stats.waits_already_done;
    trace(kWaitBegin, id_, 1);
    trace(kWaitEnd, id_, 1);
    backend_->pump(false);  // keep the next chunk moving
    return true;
  }
  Timed timed(stats.wait_s, kWaitBegin, id_, 0);
  backend_->pump(true, this);
  return true;
}

py::dict get_stats() {
  Lock lock(mpi_mutex);
  py::dict d;
  d["async"] = kAsync;
  d["wait_s"] = stats.wait_s;
  d["drain_s"] = stats.drain_s;
  d["gather_sync_s"] = stats.gather_sync_s;
  d["reduce_scatter_s"] = stats.reduce_scatter_s;
  d["allreduce_s"] = stats.allreduce_s;
  d["other_s"] = stats.other_s;
  d["gathers_async"] = stats.gathers_async;
  d["gathers_sync"] = stats.gathers_sync;
  d["chunks"] = stats.chunks;
  d["waits"] = stats.waits;
  d["waits_already_done"] = stats.waits_already_done;
  d["gather_bytes"] = stats.gather_bytes;
  return d;
}

py::list get_trace() {
  Lock lock(mpi_mutex);
  py::list out;
  for (const Event &e : events) out.append(py::make_tuple(e.t, e.kind, e.id, e.bytes));
  events.clear();
  return out;
}

void reset_stats() {
  Lock lock(mpi_mutex);
  stats = Stats{};
}

c10::intrusive_ptr<FjMpiBackend> create(const c10::intrusive_ptr<c10d::Store> &, int64_t rank, int64_t size,
                                        const std::vector<int64_t> &global_ranks, const std::string &group_id) {
  return c10::make_intrusive<FjMpiBackend>(static_cast<int>(rank), static_cast<int>(size), global_ranks,
                                           group_id);
}

void finalize() {
  Lock lock(mpi_mutex);
  int ok = 0, finalized = 0;
  MPI_Initialized(&ok);
  MPI_Finalized(&finalized);
  if (ok && !finalized) MPI_Finalize();
}

}  // namespace

PYBIND11_MODULE(_c10d, m) {
  py::module_::import("torch._C._distributed_c10d");  // registers the c10d::Backend base class
  py::class_<FjMpiBackend, c10d::Backend, c10::intrusive_ptr<FjMpiBackend>>(m, "FjMpiBackend");
  m.def("create", &create, py::arg("store"), py::arg("rank"), py::arg("size"), py::arg("global_ranks"),
        py::arg("group_id"), py::call_guard<py::gil_scoped_release>());
  m.def("finalize", &finalize, py::call_guard<py::gil_scoped_release>());
  m.def("stats", &get_stats);
  m.def("reset_stats", &reset_stats);
  m.def("trace", &get_trace);
}
