// torch.distributed backend "fjmpi": collectives over MPI for the stock wheel, which is
// built without an MPI backend. Built against Open MPI 4.0.1 headers, so the same binary
// runs on Fujitsu MPI on Fugaku. Collectives are synchronous: each call returns a
// completed Work.
//
// Data moves through two reused 8 MB staging buffers: without Fujitsu's large pages, Tofu
// registers every new buffer it sends, and torch's allocator keeps presenting new ones.
#include <mpi.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/extension.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>

namespace {

constexpr size_t kStage = 8 << 20;  // bytes per staging buffer
std::mutex mpi_mutex;               // MPI_THREAD_SERIALIZED: one caller at a time
char *stage_in = nullptr, *stage_out = nullptr;

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
  if (!stage_in) {
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

class FjMpiBackend : public c10d::Backend {
 public:
  // global_ranks: the group's members in MPI_COMM_WORLD order; empty means the world.
  FjMpiBackend(int rank, int size, const std::vector<int64_t> &global_ranks, const std::string &group_id)
      : c10d::Backend(rank, size) {
    std::lock_guard<std::mutex> lock(mpi_mutex);
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
    if (!finalized && comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
  }

  const std::string getBackendName() const override { return "fjmpi"; }

  c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor> &tensors,
                                           const c10d::AllreduceOptions &opts) override {
    for (auto &t : tensors) reduce_all(t, opts.reduceOp);
    return done(tensors);
  }

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(std::vector<at::Tensor> &tensors,
                                                     const c10d::AllreduceCoalescedOptions &opts) override {
    for (auto &t : tensors) reduce_all(t, opts.reduceOp);
    return done(tensors);
  }

  c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor> &tensors,
                                           const c10d::BroadcastOptions &opts) override {
    std::lock_guard<std::mutex> lock(mpi_mutex);
    const int root = static_cast<int>(opts.rootRank);
    for (auto &t : tensors) {
      check_tensor(t);
      char *p = static_cast<char *>(t.data_ptr());
      const size_t n = t.nbytes();
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
    for (auto &t : tensors) reduce_to(t, opts.reduceOp, static_cast<int>(opts.rootRank));
    return done(tensors);
  }

  c10::intrusive_ptr<c10d::Work> _allgather_base(at::Tensor &out, at::Tensor &in,
                                                 const c10d::AllgatherOptions &) override {
    gather_all(out, in);
    return done({out});
  }

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(std::vector<at::Tensor> &outs,
                                                                 std::vector<at::Tensor> &ins,
                                                                 const c10d::AllgatherOptions &) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: allgather_into_tensor_coalesced size mismatch");
    for (size_t i = 0; i < ins.size(); ++i) gather_all(outs[i], ins[i]);
    return done(outs);
  }

  c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>> &outs,
                                           std::vector<at::Tensor> &ins,
                                           const c10d::AllgatherOptions &) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: allgather size mismatch");
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
    scatter_reduce(out, in, opts.reduceOp);
    return done({out});
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(std::vector<at::Tensor> &outs,
                                                                 std::vector<at::Tensor> &ins,
                                                                 const c10d::ReduceScatterOptions &opts) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: reduce_scatter_tensor_coalesced size mismatch");
    for (size_t i = 0; i < ins.size(); ++i) scatter_reduce(outs[i], ins[i], opts.reduceOp);
    return done(outs);
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter(std::vector<at::Tensor> &outs,
                                                std::vector<std::vector<at::Tensor>> &ins,
                                                const c10d::ReduceScatterOptions &opts) override {
    TORCH_CHECK(outs.size() == ins.size(), "fjmpi: reduce_scatter size mismatch");
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
    std::lock_guard<std::mutex> lock(mpi_mutex);
    if (out_splits.empty() && in_splits.empty()) {
      TORCH_CHECK(in.nbytes() == out.nbytes() && in.nbytes() % size_ == 0,
                  "fjmpi: alltoall_base needs equal input and output divisible by the group size");
      const size_t n = in.nbytes() / size_;
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
    std::lock_guard<std::mutex> lock(mpi_mutex);
    check(MPI_Barrier(comm_), "MPI_Barrier");
    return done({});
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;

  void reduce_all(at::Tensor &t, const c10d::ReduceOp &op) {
    check_tensor(t);
    if (half_like(t)) {  // no MPI type: reduce in float32
      at::Tensor f = t.to(at::kFloat);
      reduce_all(f, op);
      t.copy_(f);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mpi_mutex);
      const MPI_Datatype type = mpi_type(t);
      const MPI_Op mop = mpi_op(op, t);
      const size_t es = t.element_size(), chunk = kStage / es;
      char *p = static_cast<char *>(t.data_ptr());
      const size_t n = t.numel();
      for (size_t off = 0; off < n; off += chunk) {
        const size_t cnt = std::min(chunk, n - off);
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
      std::lock_guard<std::mutex> lock(mpi_mutex);
      const MPI_Datatype type = mpi_type(t);
      const MPI_Op mop = mpi_op(op, t);
      const size_t es = t.element_size(), chunk = kStage / es;
      char *p = static_cast<char *>(t.data_ptr());
      const size_t n = t.numel();
      for (size_t off = 0; off < n; off += chunk) {
        const size_t cnt = std::min(chunk, n - off);
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
    std::lock_guard<std::mutex> lock(mpi_mutex);
    const size_t n = in.nbytes(), chunk = kStage / size_;
    TORCH_CHECK(chunk > 0, "fjmpi: group too large for the staging buffer");
    const char *src = static_cast<const char *>(in.data_ptr());
    char *dst = static_cast<char *>(out.data_ptr());
    for (size_t off = 0; off < n; off += chunk) {
      const size_t len = std::min(chunk, n - off);
      std::memcpy(stage_in, src + off, len);
      check(MPI_Allgather(stage_in, static_cast<int>(len), MPI_BYTE, stage_out, static_cast<int>(len), MPI_BYTE,
                          comm_), "MPI_Allgather");
      for (int r = 0; r < size_; ++r) std::memcpy(dst + r * n + off, stage_out + r * len, len);
    }
  }

  // out = reduction over ranks of their in[rank * out.numel() : (rank + 1) * out.numel()].
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
    {
      std::lock_guard<std::mutex> lock(mpi_mutex);
      const MPI_Datatype type = mpi_type(out);
      const MPI_Op mop = mpi_op(op, out);
      const size_t es = out.element_size(), n = out.numel(), chunk = kStage / (es * size_);
      TORCH_CHECK(chunk > 0, "fjmpi: group too large for the staging buffer");
      const char *src = static_cast<const char *>(in.data_ptr());
      char *dst = static_cast<char *>(out.data_ptr());
      for (size_t off = 0; off < n; off += chunk) {
        const size_t cnt = std::min(chunk, n - off);
        for (int r = 0; r < size_; ++r) std::memcpy(stage_in + r * cnt * es, src + (r * n + off) * es, cnt * es);
        check(MPI_Reduce_scatter_block(stage_in, stage_out, static_cast<int>(cnt), type, mop, comm_),
              "MPI_Reduce_scatter_block");
        std::memcpy(dst + off * es, stage_out, cnt * es);
      }
    }
    if (op.op_ == c10d::ReduceOp::AVG) out.div_(size_);
  }
};

c10::intrusive_ptr<FjMpiBackend> create(const c10::intrusive_ptr<c10d::Store> &, int64_t rank, int64_t size,
                                        const std::vector<int64_t> &global_ranks, const std::string &group_id) {
  return c10::make_intrusive<FjMpiBackend>(static_cast<int>(rank), static_cast<int>(size), global_ranks,
                                           group_id);
}

void finalize() {
  std::lock_guard<std::mutex> lock(mpi_mutex);
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
}
