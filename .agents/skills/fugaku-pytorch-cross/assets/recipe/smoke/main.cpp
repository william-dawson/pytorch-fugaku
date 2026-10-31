#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

int main() {
  auto pg=c10d::ProcessGroupMPI::createProcessGroupMPI();
  const int rank=pg->getRank(),size=pg->getSize();
  at::set_num_threads(std::getenv("OMP_NUM_THREADS")?std::atoi(std::getenv("OMP_NUM_THREADS")):12);
  int actual_threads=0;
  #pragma omp parallel
  {
    #pragma omp single
    actual_threads=omp_get_num_threads();
  }
  TORCH_CHECK(actual_threads==at::get_num_threads(),"OpenMP thread count mismatch");
  auto options=at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
  auto x=at::ones({32,32},options).set_requires_grad(true);
  auto y=at::mm(x,at::ones({32,32},options));
  torch::autograd::backward({y.sum()});
  TORCH_CHECK(at::all(y.eq(32)).item<bool>(),"GEMM failed");
  TORCH_CHECK(at::all(x.grad().eq(32)).item<bool>(),"autograd failed");
  auto solution=at::linalg_solve(at::eye(3,options)*2,at::ones({3,1},options)*4);
  TORCH_CHECK(at::all((solution-2).abs().lt(1e-10)).item<bool>(),"LAPACK failed");
  std::vector<at::Tensor> tensors{at::full({4},rank+1,options)};
  pg->allreduce(tensors)->wait();
  TORCH_CHECK(at::all(tensors[0].eq(size*(size+1)/2)).item<bool>(),"ProcessGroupMPI allreduce failed");
  char host[MPI_MAX_PROCESSOR_NAME]={};int length=0,provided=0;
  MPI_Get_processor_name(host,&length);MPI_Query_thread(&provided);
  std::vector<char> hosts(size*MPI_MAX_PROCESSOR_NAME);
  MPI_Allgather(host,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,hosts.data(),MPI_MAX_PROCESSOR_NAME,MPI_CHAR,MPI_COMM_WORLD);
  std::set<std::string> unique;
  for(int i=0;i<size;++i)unique.insert(hosts.data()+i*MPI_MAX_PROCESSOR_NAME);
  TORCH_CHECK(size>=2 && unique.size()>=2,"Validation requires two compute nodes");
  std::printf("rank=%d host=%s mpi_threads=%d omp_threads=%d GEMM=PASS LAPACK=PASS autograd=PASS ProcessGroupMPI=PASS OpenMP=PASS\n",rank,host,provided,actual_threads);
  pg.reset();
  return 0;
}
