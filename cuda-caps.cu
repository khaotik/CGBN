#include <stdio.h>
#include <cuda_runtime.h>
int main(){cudaDeviceProp p;if(cudaSuccess!=cudaGetDeviceProperties(&p,0))return 1;printf("%d%d",p.major,p.minor);return 0;}
