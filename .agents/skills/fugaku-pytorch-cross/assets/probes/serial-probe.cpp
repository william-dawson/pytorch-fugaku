#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

extern "C" {
void sgemm_(const char*, const char*, const int*, const int*, const int*,
            const float*, const float*, const int*, const float*, const int*,
            const float*, float*, const int*);
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
void dgesv_(const int*, const int*, double*, const int*, int*, double*, const int*, int*);
}

static void origin(const char* name, void* symbol) {
    Dl_info info{};
    std::printf("library %s=%s\n", name,
                dladdr(symbol, &info) && info.dli_fname ? info.dli_fname : "unknown");
}

template<class T> bool gemm(bool transpose) {
    const int m=37, n=29, k=41;
    const int lda=transpose ? k : m, ldb=k, ldc=m;
    const char ta=transpose ? 'T' : 'N', tb='N';
    const T alpha=T(0.75), beta=T(-0.25);
    std::vector<T> a(m*k), b(k*n), c(m*n), expected(m*n);
    for (int i=0; i<m*k; ++i) a[i]=T((i%17)-8)/T(16);
    for (int i=0; i<k*n; ++i) b[i]=T((i%13)-6)/T(8);
    for (int i=0; i<m*n; ++i) c[i]=expected[i]=T((i%7)-3)/T(4);
    for (int j=0; j<n; ++j) for (int i=0; i<m; ++i) {
        T sum=0;
        for (int p=0; p<k; ++p) sum+=(transpose ? a[p+i*lda] : a[i+p*lda])*b[p+j*ldb];
        expected[i+j*ldc]=alpha*sum+beta*expected[i+j*ldc];
    }
    if constexpr (sizeof(T)==sizeof(float))
        sgemm_(&ta,&tb,&m,&n,&k,&alpha,a.data(),&lda,b.data(),&ldb,&beta,c.data(),&ldc);
    else
        dgemm_(&ta,&tb,&m,&n,&k,&alpha,a.data(),&lda,b.data(),&ldb,&beta,c.data(),&ldc);
    for (int i=0; i<m*n; ++i)
        if (!std::isfinite(c[i]) || std::abs(c[i]-expected[i])>T(1e-5)*(T(1)+std::abs(expected[i]))) return false;
    return true;
}

int main(int argc, char** argv) {
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0, size=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    char host[MPI_MAX_PROCESSOR_NAME]={};
    int host_length=0;
    MPI_Get_processor_name(host,&host_length);
    std::vector<char> hosts(size*MPI_MAX_PROCESSOR_NAME);
    MPI_Allgather(host,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,hosts.data(),MPI_MAX_PROCESSOR_NAME,MPI_CHAR,MPI_COMM_WORLD);
    std::set<std::string> unique;
    for (int i=0; i<size; ++i) unique.insert(hosts.data()+i*MPI_MAX_PROCESSOR_NAME);
    int threads=0, sum=0;
    #pragma omp parallel reduction(+:sum)
    {
        #pragma omp single
        threads=omp_get_num_threads();
        sum+=1;
    }
    int local=rank+1, total=0;
    MPI_Allreduce(&local,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    bool fp32=gemm<float>(false) && gemm<float>(true);
    bool fp64=gemm<double>(false) && gemm<double>(true);
    // Column-major A, with a known solution x=[1,2,3].
    int n=3, nrhs=1, info=0, pivots[3]={};
    double a[]={4,1,2,1,5,1,2,1,6}, b[]={12,14,22};
    dgesv_(&n,&nrhs,a,&n,pivots,b,&n,&info);
    bool lapack=info==0;
    for(int i=0;i<3;++i) lapack=lapack && std::isfinite(b[i]) && std::abs(b[i]-(i+1))<1e-12;
    bool ok=provided>=MPI_THREAD_FUNNELED && size>=2 && unique.size()>=2 &&
            total==size*(size+1)/2 && threads==omp_get_max_threads() && sum==threads && fp32 && fp64 && lapack;
    std::printf("rank=%d host=%s threads=%d sgemm=%s dgemm=%s dgesv=%s local=%s\n",
                rank,host,threads,fp32?"PASS":"FAIL",fp64?"PASS":"FAIL",lapack?"PASS":"FAIL",ok?"PASS":"FAIL");
    if(rank==0) {
        std::printf("compiler=%s\n",__clang_version__);
        char version[MPI_MAX_LIBRARY_VERSION_STRING]={}; int length=0;
        MPI_Get_library_version(version,&length);
        std::printf("MPI version=%.*s\n",length,version);
        origin("MPI",reinterpret_cast<void*>(&MPI_Init_thread));
        origin("SGEMM",reinterpret_cast<void*>(&sgemm_));
        origin("DGEMM",reinterpret_cast<void*>(&dgemm_));
        origin("DGESV",reinterpret_cast<void*>(&dgesv_));
        origin("OpenMP",reinterpret_cast<void*>(&omp_get_max_threads));
        std::ifstream maps("/proc/self/maps"); std::string line; std::set<std::string> libraries;
        while(std::getline(maps,line)) {
            auto start=line.find('/');
            if(start!=std::string::npos && (line.find("libomp")!=std::string::npos ||
               line.find("libgomp")!=std::string::npos || line.find("libfjomp")!=std::string::npos)) libraries.insert(line.substr(start));
        }
        for(const auto& lib:libraries) std::printf("mapped OpenMP=%s\n",lib.c_str());
        std::printf("ranks=%d distinct_hosts=%zu allreduce=%d expected=%d\n",size,unique.size(),total,size*(size+1)/2);
    }
    int local_ok=ok?1:0, all_ok=0;
    MPI_Allreduce(&local_ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(rank==0) std::printf("OVERALL=%s\n",all_ok?"PASS":"FAIL");
    MPI_Finalize();
    return all_ok?0:1;
}
