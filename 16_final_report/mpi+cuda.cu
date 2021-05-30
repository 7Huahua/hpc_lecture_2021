#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

__global__ void matrix(float *a,float *b,float *c,int N, int offset,int size){
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j <N/size){
    for (int i=0; i<N/size; i++)
      for (int k=0; k<N; k++)
        c[N*i+j+offset] += a[N*i+k] * b[N/size*k+j];
  }
}

int main(int argc, char** argv) {
  int size, rank;
  int gpusize;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaGetDeviceCount(&gpusize);
  cudaSetDevice(rank % gpusize);
  const int N = 256;
  const int M = 256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  float *subA, *subB, *subC,*recv;
  subA = (float *)malloc(N*N/size*sizeof(float));
  subB = (float *)malloc(N*N/size*sizeof(float));
  subC = (float *)malloc(N*N/size*sizeof(float));
  recv = (float *)malloc(N*N/size*sizeof(float));

  float *a;
  float *b;
  float *c;
  cudaMalloc(&a, N*N/size*sizeof(float));
  cudaMalloc(&b, N*N/size*sizeof(float));
  cudaMalloc(&c, N*N/size*sizeof(float));
  cudaDeviceEnablePeerAccess(rank%gpusize, 0);

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
  int offset = N/size*rank;

  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  cudaMemcpy(a,subA,N*N/size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(b,subB,N*N/size*sizeof(float),cudaMemcpyHostToDevice);
  
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;
  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank+irank) % size);
    matrix<<<(N/size+M-1)/M,M>>>(a,b,c,N,offset,size);
    cudaDeviceSynchronize();

    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
   MPI_Request request[2];
   MPI_Isend(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
   MPI_Irecv(&recv[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
   MPI_Waitall(2, request, MPI_STATUS_IGNORE);
   for (int i=0; i<N*N/size; i++)
     subB[i] = recv[i];
    cudaMemcpy(b,subB,N*N/size*sizeof(float),cudaMemcpyHostToDevice);
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  cudaMemcpy(subC,c,N*N/size*sizeof(float),cudaMemcpyDeviceToHost);
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

 for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("size : %d\n",size);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  free(subA);
  free(subB);
  free(subC);
  free(recv);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  MPI_Finalize();
}