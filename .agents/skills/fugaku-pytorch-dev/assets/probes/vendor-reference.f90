program vendor_reference
  use mpi
  use omp_lib
  use iso_fortran_env, only: real32, real64
  implicit none
  integer :: ierr, rank, ranks, provided, threads, participants, output, n, case_id
  character(len=256) :: argument, path
  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, ranks, ierr)
  call get_command_argument(1, argument)
  read(argument, *) threads
  call get_command_argument(2, path)
  call omp_set_dynamic(.false.)
  call omp_set_num_threads(threads)
  participants = 0
!$omp parallel reduction(+:participants)
  participants = participants + 1
!$omp end parallel
  if (participants /= threads .or. ranks*threads /= 48) then
    call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
  end if
  write(*,*) 'TEAM rank=', rank, 'ranks=', ranks, 'threads=', participants
  if (rank == 0) then
    open(newunit=output, file=trim(path), status='replace')
    write(output,'(A)') 'dtype,ranks,threads,n,median_seconds,node_gflops,correct'
  end if
  do case_id = 1, 2
    n = 4096 * case_id
    call bench_s(n)
    call bench_d(n)
  end do
  if (rank == 0) then
    close(output)
    write(*,*) 'OVERALL=PASS'
  end if
  call MPI_Finalize(ierr)
contains
  subroutine bench_s(n)
    integer, intent(in) :: n
    real(real32), allocatable :: a(:,:), b(:,:), c(:,:)
    real(real64) :: start, elapsed, worst, samples(7), temporary, median
    integer :: i, j, trial, warmup, ok, all_ok
    allocate(a(n,n), b(n,n), c(n,n))
!$omp parallel do schedule(static) private(i)
    do j = 1, n
      do i = 1, n
        a(i,j) = 1.0_real32
        b(i,j) = 0.5_real32
        c(i,j) = 0.0_real32
      end do
    end do
!$omp end parallel do
    do warmup = 1, 2
      call sgemm('N','N',n,n,n,1.0_real32,a,n,b,n,0.0_real32,c,n)
    end do
    ok = 1
    if (maxval(abs(c-real(n,real32)*0.5_real32)) > 1.0e-5_real32) ok = 0
    call MPI_Allreduce(ok, all_ok, 1, MPI_INTEGER, MPI_MIN, MPI_COMM_WORLD, ierr)
    if (all_ok /= 1) call MPI_Abort(MPI_COMM_WORLD, 4, ierr)
    do trial = 1, 7
      call MPI_Barrier(MPI_COMM_WORLD, ierr)
      start = MPI_Wtime()
      call sgemm('N','N',n,n,n,1.0_real32,a,n,b,n,0.0_real32,c,n)
      elapsed = MPI_Wtime() - start
      call MPI_Reduce(elapsed, worst, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
      samples(trial) = worst
    end do
    if (rank == 0) then
      do i = 2, 7
        temporary = samples(i)
        j = i-1
        do while (j >= 1)
          if (samples(j) <= temporary) exit
          samples(j+1) = samples(j)
          j = j-1
        end do
        samples(j+1) = temporary
      end do
      median = samples(4)
      write(*,*) 'fp32', ' n=', n, ' ranks=', ranks, ' threads=', threads, &
        ' seconds=', median, ' node_GFLOPs=', 2.0_real64*ranks*real(n,real64)**3/median/1.0e9_real64
      write(output,'(A,3(",",I0),2(",",ES24.16),",1")') &
        'fp32', ranks, threads, n, median, 2.0_real64*ranks*real(n,real64)**3/median/1.0e9_real64
      flush(output)
    end if
    deallocate(a,b,c)
  end subroutine bench_s
  subroutine bench_d(n)
    integer, intent(in) :: n
    real(real64), allocatable :: a(:,:), b(:,:), c(:,:)
    real(real64) :: start, elapsed, worst, samples(7), temporary, median
    integer :: i, j, trial, warmup, ok, all_ok
    allocate(a(n,n), b(n,n), c(n,n))
!$omp parallel do schedule(static) private(i)
    do j = 1, n
      do i = 1, n
        a(i,j) = 1.0_real64
        b(i,j) = 0.5_real64
        c(i,j) = 0.0_real64
      end do
    end do
!$omp end parallel do
    do warmup = 1, 2
      call dgemm('N','N',n,n,n,1.0_real64,a,n,b,n,0.0_real64,c,n)
    end do
    ok = 1
    if (maxval(abs(c-real(n,real64)*0.5_real64)) > 1.0e-5_real64) ok = 0
    call MPI_Allreduce(ok, all_ok, 1, MPI_INTEGER, MPI_MIN, MPI_COMM_WORLD, ierr)
    if (all_ok /= 1) call MPI_Abort(MPI_COMM_WORLD, 4, ierr)
    do trial = 1, 7
      call MPI_Barrier(MPI_COMM_WORLD, ierr)
      start = MPI_Wtime()
      call dgemm('N','N',n,n,n,1.0_real64,a,n,b,n,0.0_real64,c,n)
      elapsed = MPI_Wtime() - start
      call MPI_Reduce(elapsed, worst, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
      samples(trial) = worst
    end do
    if (rank == 0) then
      do i = 2, 7
        temporary = samples(i)
        j = i-1
        do while (j >= 1)
          if (samples(j) <= temporary) exit
          samples(j+1) = samples(j)
          j = j-1
        end do
        samples(j+1) = temporary
      end do
      median = samples(4)
      write(*,*) 'fp64', ' n=', n, ' ranks=', ranks, ' threads=', threads, &
        ' seconds=', median, ' node_GFLOPs=', 2.0_real64*ranks*real(n,real64)**3/median/1.0e9_real64
      write(output,'(A,3(",",I0),2(",",ES24.16),",1")') &
        'fp64', ranks, threads, n, median, 2.0_real64*ranks*real(n,real64)**3/median/1.0e9_real64
      flush(output)
    end if
    deallocate(a,b,c)
  end subroutine bench_d
end program vendor_reference
