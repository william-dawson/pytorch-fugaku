#define OMPI_SKIP_MPICXX 1
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <mpi.h>
#include <omp.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

extern "C" {
void sgemm_(const char*,const char*,const int*,const int*,const int*,const float*,const float*,const int*,const float*,const int*,const float*,float*,const int*);
void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,const double*,const int*,const double*,const int*,const double*,double*,const int*);
}

static void invoke_gemm(const char* ta,const char* tb,const int* m,const int* n,const int* k,
    const float* alpha,const float* a,const int* lda,const float* b,const int* ldb,
    const float* beta,float* c,const int* ldc) {
    sgemm_(ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

static void invoke_gemm(const char* ta,const char* tb,const int* m,const int* n,const int* k,
    const double* alpha,const double* a,const int* lda,const double* b,const int* ldb,
    const double* beta,double* c,const int* ldc) {
    dgemm_(ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

static double cpu_time() {
    timespec value{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&value);
    return value.tv_sec+value.tv_nsec*1e-9;
}
static std::string origin(void* symbol) {
    Dl_info info{};
    return dladdr(symbol,&info) && info.dli_fname ? info.dli_fname : "unknown";
}
static std::vector<int> cpu_mask() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if(sched_getaffinity(0,sizeof(mask),&mask)!=0) MPI_Abort(MPI_COMM_WORLD,10);
    std::vector<int> result(CPU_SETSIZE,0);
    for(int i=0;i<CPU_SETSIZE;++i) result[i]=CPU_ISSET(i,&mask)?1:0;
    return result;
}
static void runtime_maps(int rank) {
    std::ifstream input("/proc/self/maps"); std::string line; std::set<std::string> paths;
    while(std::getline(input,line)) {
        auto start=line.find('/');
        if(start!=std::string::npos && (line.find("libomp")!=std::string::npos || line.find("libfjomp")!=std::string::npos || line.find("libgomp")!=std::string::npos)) paths.insert(line.substr(start));
    }
    for(const auto& path:paths) std::printf("RUNTIME rank=%d path=%s\n",rank,path.c_str());
}

struct Shape { int m,n,k; char transb; };
static std::vector<int> placement_cpus;

template<class T> static auto allocate(size_t count) {
    void* pointer=nullptr;
    if(posix_memalign(&pointer,256,count*sizeof(T))!=0) MPI_Abort(MPI_COMM_WORLD,11);
    return std::unique_ptr<T,decltype(&std::free)>(static_cast<T*>(pointer),&std::free);
}

template<class T> static void initialize(T* a,size_t na,T* b,size_t nb,T* c,size_t nc) {
    // First-touch pages across each rank's cores/CMGs, outside timed regions.
    std::vector<std::thread> workers;
    std::atomic<bool> bound{true};
    const size_t nt=placement_cpus.size();
    for(size_t t=0;t<nt;++t) workers.emplace_back([&,t]() {
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(placement_cpus[t],&mask);
        if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)!=0) bound=false;
        for(size_t i=na*t/nt;i<na*(t+1)/nt;++i) a[i]=T((int(i%17)-8))/T(16);
        for(size_t i=nb*t/nt;i<nb*(t+1)/nt;++i) b[i]=T((int(i%13)-6))/T(16);
        for(size_t i=nc*t/nt;i<nc*(t+1)/nt;++i) c[i]=T(0);
    });
    for(auto& worker:workers) worker.join();
    if(!bound) MPI_Abort(MPI_COMM_WORLD,12);
}

template<class T> static bool benchmark(const Shape& shape,int rank,int size,int threads,FILE* csv) {
    const int m=shape.m,n=shape.n,k=shape.k,lda=m,ldb=shape.transb=='N'?k:n,ldc=m;
    const char ta='N',tb=shape.transb;
    const T alpha=T(1),beta=T(0);
    auto a=allocate<T>(size_t(m)*k),b=allocate<T>(size_t(n)*k),c=allocate<T>(size_t(m)*n);
    initialize(a.get(),size_t(m)*k,b.get(),size_t(n)*k,c.get(),size_t(m)*n);
    auto gemm=[&]() {
        invoke_gemm(&ta,&tb,&m,&n,&k,&alpha,a.get(),&lda,b.get(),&ldb,&beta,c.get(),&ldc);
    };
    gemm(); gemm();
    bool correct=true;
    // Deterministic samples across the output, checked with FP64 accumulation.
    for(int sample=0;sample<64;++sample) {
        int row=(sample*7919+rank*17)%m,col=(sample*3571+rank*29)%n;
        double expected=0;
        for(int p=0;p<k;++p) expected+=double(a.get()[row+size_t(p)*lda])*double(tb=='N'?b.get()[p+size_t(col)*ldb]:b.get()[col+size_t(p)*ldb]);
        double got=double(c.get()[row+size_t(col)*ldc]);
        if(!std::isfinite(got) || std::abs(got-expected)>(sizeof(T)==4?1e-4:1e-11)*(1+std::abs(expected))) correct=false;
    }
    std::vector<double> worst_times,active_cores;
    // Seven synchronized trials; node throughput uses the slowest rank.
    for(int trial=0;trial<7;++trial) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double cpu_start=cpu_time(),start=MPI_Wtime();
        gemm();
        const double elapsed=MPI_Wtime()-start,cpu_elapsed=cpu_time()-cpu_start;
        double worst=0,cores=cpu_elapsed/elapsed,total_cores=0;
        MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&cores,&total_cores,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(rank==0) {worst_times.push_back(worst);active_cores.push_back(total_cores);}
    }
    int local=correct?1:0,all_correct=0;
    MPI_Allreduce(&local,&all_correct,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(rank==0) {
        std::sort(worst_times.begin(),worst_times.end());
        std::sort(active_cores.begin(),active_cores.end());
        double flops=2.0*m*n*k;
        std::printf("RESULT dtype=fp%zu ranks=%d threads=%d m=%d n=%d k=%d tb=%c median_ms=%.4f min_ms=%.4f max_ms=%.4f node_GFLOPs=%.2f cpu_cores=%.2f correct=%s\n",
            sizeof(T)*8,size,threads,m,n,k,tb,1000*worst_times[3],1000*worst_times.front(),1000*worst_times.back(),size*flops/worst_times[3]/1e9,active_cores[3],all_correct?"PASS":"FAIL");
        std::fprintf(csv,"fp%zu,%d,%d,%d,%d,%d,%c,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
            sizeof(T)*8,size,threads,m,n,k,tb,worst_times[3],worst_times.front(),worst_times.back(),size*flops/worst_times[3]/1e9,active_cores[3],all_correct);
        std::fflush(csv);
    }
    return all_correct;
}

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,size=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc!=3) {if(rank==0) std::fprintf(stderr,"usage: bench THREADS OUTPUT.csv\n");MPI_Abort(MPI_COMM_WORLD,2);}
    int threads=std::atoi(argv[1]);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm);
    int local_size=0; MPI_Comm_size(local_comm,&local_size);
    bool valid=provided>=MPI_THREAD_FUNNELED && threads>0 && size==local_size && size*threads==48;
    auto mask=cpu_mask();
    std::vector<int> all_masks(size*CPU_SETSIZE);
    MPI_Allgather(mask.data(),CPU_SETSIZE,MPI_INT,all_masks.data(),CPU_SETSIZE,MPI_INT,MPI_COMM_WORLD);
    int allowed=0,overlap=0;
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu) {
        allowed+=mask[cpu];
        if(mask[cpu]) placement_cpus.push_back(cpu);
        if(mask[cpu]) for(int other=0;other<size;++other) if(other!=rank && all_masks[other*CPU_SETSIZE+cpu]) ++overlap;
    }
    // Report and reject overlapping or undersized rank bindings.
    valid=valid && allowed==threads && overlap==0;
    char host[MPI_MAX_PROCESSOR_NAME]={};int length=0;MPI_Get_processor_name(host,&length);
    std::printf("PLACEMENT rank=%d host=%s allowed=%d expected=%d overlap=%d cpus=",rank,host,allowed,threads,overlap);
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(mask[cpu]) std::printf("%d,",cpu);
    std::printf("\n");
    omp_set_dynamic(0); omp_set_num_threads(threads);
    int team=0,participants=0;
#ifdef _OPENMP
    #pragma omp parallel reduction(+:participants)
    {
        #pragma omp single
        team=omp_get_num_threads();
        participants+=1;
    }
    valid=valid && team==threads && participants==threads;
#endif
    valid=valid && omp_get_max_threads()==threads;
    std::printf("OPENMP rank=%d requested=%d max=%d compiled_parallel=%d team=%d participants=%d api=%s\n",rank,threads,omp_get_max_threads(),
#ifdef _OPENMP
        1,
#else
        0,
#endif
        team,participants,origin(reinterpret_cast<void*>(&omp_get_max_threads)).c_str());
    if(rank==0) {
        std::printf("LIBRARY sgemm=%s dgemm=%s MPI=%s\n",origin(reinterpret_cast<void*>(&sgemm_)).c_str(),origin(reinterpret_cast<void*>(&dgemm_)).c_str(),origin(reinterpret_cast<void*>(&MPI_Init_thread)).c_str());
    }
    int local_valid=valid?1:0,all_valid=0;
    MPI_Allreduce(&local_valid,&all_valid,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!all_valid) {std::printf("PRECHECK=FAIL rank=%d\n",rank);MPI_Finalize();return 3;}
    FILE* csv=rank==0?std::fopen(argv[2],"w"):nullptr;
    if(rank==0 && !csv) MPI_Abort(MPI_COMM_WORLD,4);
    if(rank==0) std::fprintf(csv,"dtype,ranks,threads,m,n,k,transb,median_seconds,min_seconds,max_seconds,node_gflops,process_cpu_cores,correct\n");
    const Shape shapes[]={{1024,1024,1024,'N'},{4096,4096,4096,'N'},{8192,8192,8192,'N'},{3840,1536,512,'N'},{2048,2048,8192,'T'}};
    bool ok=true;
    for(const auto& shape:shapes) {
        ok=benchmark<float>(shape,rank,size,threads,csv) && ok;
        ok=benchmark<double>(shape,rank,size,threads,csv) && ok;
    }
    runtime_maps(rank);
    if(rank==0) {std::fclose(csv);std::printf("OVERALL=%s\n",ok?"PASS":"FAIL");}
    MPI_Comm_free(&local_comm);MPI_Finalize();return ok?0:1;
}
