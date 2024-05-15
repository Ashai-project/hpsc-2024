#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void count_key(int *bucket, int *key, int n)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;
  atomicAdd(&bucket[key[tid]], 1);
}

__global__ void culc_offset(int *bucket, int *key_offset, int range)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= range)
    return;
  for (int i = 0; i < range; i++)
  {
    if (tid + i >= range)
      return;
    atomicAdd(&key_offset[tid + i], bucket[i]);
  }
}

__global__ void set_key(int *key_offset, int *key, int n, int range)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int value = tid;
  for (int i = 0; i < range; i++)
  {
    if (key_offset[i] > tid)
    {
      value = i;
      break;
    }
  }
  if (tid >= n)
    return;
  key[tid] = value;
}

int main()
{
  int n = 50;
  const int M = 1024;
  int range = 5;
  // std::vector<int> key(n);
  int *key_h, *bucket_d, *key_d, *key_offset;
  cudaMallocHost((void **)&key_h, sizeof(int) * n);
  for (int i = 0; i < n; i++)
  {
    key_h[i] = rand() % range;
    printf("%d ", key_h[i]);
  }
  printf("\n");

  // std::vector<int> bucket(range);
  // for (int i=0; i<range; i++) {
  //   bucket[i] = 0;
  // }
  // for (int i=0; i<n; i++) {
  //   bucket[key[i]]++;
  // }
  // for (int i=0, j=0; i<range; i++) {
  //   for (; bucket[i]>0; bucket[i]--) {
  //     key[j++] = i;
  //   }
  // }

  cudaMalloc((void **)&bucket_d, sizeof(int) * range);
  cudaMalloc((void **)&key_offset, sizeof(int) * range);
  cudaMalloc((void **)&key_d, sizeof(int) * n);
  cudaMemcpy(key_d, key_h, sizeof(int) * n, cudaMemcpyHostToDevice);
  cudaMemset(bucket_d, 0, sizeof(int) * range);
  cudaMemset(key_offset, 0, sizeof(int) * range);
  count_key<<<(n + M - 1) / M, M>>>(bucket_d, key_d, n);
  culc_offset<<<(range + M - 1) / M, M>>>(bucket_d, key_offset, range);
  set_key<<<(n + M - 1) / M, M>>>(key_offset, key_d, n, range);
  cudaMemcpy(key_h, key_d, sizeof(int) * n, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++)
  {
    printf("%d ", key_h[i]);
  }
  printf("\n");
  cudaFree(bucket_d);
  cudaFree(key_offset);
  cudaFree(key_d);
  cudaFreeHost(key_h);
}
