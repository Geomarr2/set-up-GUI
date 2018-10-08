#include "cufft.h"
#include "cuda.h"
#include "cuComplex.h"
#include <vector>
#include <sstream>
#include <stdexcept>
#include <iostream>

#define CUDA_ERROR_CHECK(ans) { cuda_assert_success((ans), __FILE__, __LINE__); }
#define CUFFT_ERROR_CHECK(ans) { cufft_assert_success((ans), __FILE__, __LINE__); }
#define LOG(str) { /*std::cout << (str) << std::endl;*/}
#define MAX_THREADS 1024

/**
 * @brief Function that raises an error on receipt of any cudaError_t
 *  value that is not cudaSuccess
 */
inline void cuda_assert_success(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
    {
      std::stringstream error_msg;
              error_msg << "CUDA failed with error: "
			<< cudaGetErrorString(code) << std::endl
			<< "File: " << file << std::endl
			<< "Line: " << line << std::endl;
	      throw std::runtime_error(error_msg.str());
    }
}

inline void cufft_assert_success(cufftResult code, const char *file, int line)
{
  if (code != CUFFT_SUCCESS)
    {
      std::stringstream error_msg;
              error_msg << "CUFFT failed with error: "
			<< code << std::endl
			<< "File: " << file << std::endl
			<< "Line: " << line << std::endl;
	      throw std::runtime_error(error_msg.str());
    }
}

struct Plan
{
  cufftHandle plan;
  cufftComplex* input;
  cufftComplex* output;
  float* detected;
  int size;
  int batch;
  int input_bytes;
  int output_bytes;
};

__global__
void detect_and_integrate(cufftComplex* in, float* out, int nbins, int batch)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = nbins * batch;
  float xx = 0.0f, yy = 0.0f;
  for (int ii=idx; ii<size; ii+=nbins)
    {
      cufftComplex val = in[ii];
      xx += val.x * val.x;
      yy += val.y * val.y;
    }
  out[idx] = xx + yy;
}

extern "C" {

  void init(Plan* plan)
  {
    CUDA_ERROR_CHECK(cudaSetDevice(0));
    plan->input_bytes = (plan->size*plan->batch)*sizeof(cufftComplex);
    plan->output_bytes = (plan->size)*sizeof(float);
    LOG("Device set");
    CUDA_ERROR_CHECK(cudaMalloc(&(plan->input),plan->input_bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&(plan->output),plan->input_bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&(plan->detected),plan->output_bytes));
    LOG("All memory buffers generated");
    CUFFT_ERROR_CHECK(cufftPlan1d(&(plan->plan),plan->size,CUFFT_C2C,plan->batch));
    LOG("Plan generated");
    LOG(plan->input);
    LOG(plan->output);
    LOG(plan->detected);
  }

  void execute(Plan* plan, cufftComplex* in, float* out)
  {
    int nthreads_per_block = std::min(plan->size, MAX_THREADS);
    int nblocks = plan->size / nthreads_per_block;
    LOG("Copy to device");
    LOG(plan->input);
    LOG(plan->output);
    LOG(plan->detected);
    LOG(in);
    LOG(out);
    CUDA_ERROR_CHECK(cudaMemcpy(plan->input, in, plan->input_bytes, cudaMemcpyHostToDevice));
    LOG("Done... executing FFT");
    CUFFT_ERROR_CHECK(cufftExecC2C(plan->plan, plan->input, plan->output, CUFFT_FORWARD));
    LOG("Done... detecting");
    detect_and_integrate<<<nblocks,nthreads_per_block>>>(plan->output, plan->detected, plan->size, plan->batch);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    LOG("Done... copying back to host");
    CUDA_ERROR_CHECK(cudaMemcpy(out, plan->detected, plan->output_bytes, cudaMemcpyDeviceToHost));
    LOG("Done.");
  }

  void deinit(Plan* plan)
  {
    CUDA_ERROR_CHECK(cudaFree(plan->input));
    CUDA_ERROR_CHECK(cudaFree(plan->output));
    CUDA_ERROR_CHECK(cudaFree(plan->detected));
    CUFFT_ERROR_CHECK(cufftDestroy(plan->plan));
  }

} //extern "C"

int main()
{
  Plan plan;
  plan.size = 523392;
  plan.batch = 256;
  cufftComplex* input;
  float* output;
  CUDA_ERROR_CHECK(cudaMallocHost(&input,plan.size*plan.batch*sizeof(cufftComplex)));
  CUDA_ERROR_CHECK(cudaMallocHost(&output,plan.size*sizeof(float)));
  init(&plan);
  execute(&plan,input,output);
  deinit(&plan);
  CUDA_ERROR_CHECK(cudaFreeHost(input));
  CUDA_ERROR_CHECK(cudaFreeHost(output));
}