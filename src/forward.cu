/* Attention forward pass.

version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
./forward 1

*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>
#include <assert.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>


// ----------------------------------------------------------------------------
// helper functions
template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// generate random tensor
float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

// save a tensor as an image with a colormap
void save_tensor_as_image(const float* tensor, int B, int T, int C, const char* filename) {
    // Assuming the tensor is (B, T, C), use the first batch (B=0) and first channel (C=0) for visualization
    int width = T;  // Sequence length
    int height = C; // Embedding dimension

    // Create a single 2D image for the first batch and channel
    std::vector<float> single_channel(width * height);
    for (int i = 0; i < width * height; ++i) {
        single_channel[i] = tensor[i];
    }

    // Normalize the tensor values to [0, 255] for visualization
    auto [min_val, max_val] = std::minmax_element(single_channel.begin(), single_channel.end());
    float min_value = *min_val;
    float max_value = *max_val;

    std::vector<uint8_t> normalized_data(width * height);
    for (int i = 0; i < width * height; ++i) {
        normalized_data[i] = static_cast<uint8_t>(255.0f * (single_channel[i] - min_value) / (max_value - min_value));
    }

    // Convert to OpenCV Mat
    cv::Mat grayscale_img(height, width, CV_8UC1, normalized_data.data());

    // Apply a colormap (similar to "bwr" in Python)
    cv::Mat colored_img;
    cv::applyColorMap(grayscale_img, colored_img, cv::COLORMAP_JET); // Use COLORMAP_JET as a substitute for "bwr"

    // Save the image
    cv::imwrite(filename, colored_img);
}


// ----------------------------------------------------------------------------
// GPU kernels

__global__ void attention_query_key_kernel1(float* preatt, const float* inp,
                                           int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        int t2 = idx % T;
        int t = (idx / T) % T;
        if (t2 > t) {
            // autoregressive mask
            preatt[idx] = -INFINITY;
            return;
        }
        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int C3 = C*3;
        int hs = C / NH; // head size
        const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }
        val *= 1.0 / sqrtf(hs);

        preatt[idx] = val;
    }
}

__global__ void attention_softmax_kernel1(float* att, const float* preatt,
                                         int B, int T, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
        float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        // find maxval
        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 <= t; t2++) {
            if (preatt_bth[t2] > maxval) {
                maxval = preatt_bth[t2];
            }
        }

        // calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                att_bth[t2] *= expsum_inv;
            } else {
                // causal attention mask. not strictly necessary to set to zero here
                // only doing this explicitly for debugging and checking to PyTorch
                att_bth[t2] = 0.0f;
            }
        }
    }
}

__global__ void attention_value_kernel1(float* out, const float* att, const float* inp,
                                       int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C*3;
        int hs = C / NH; // head size

        float* out_bth = out + b * T * C + t * C + h * hs;
        const float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 <= t; t2++) {
           const  float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void forward1(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // attention calculation
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    attention_query_key_kernel1<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH);
    // softmax and value accumulation
    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    attention_softmax_kernel1<<<num_blocks, block_size>>>(att, preatt, B, T, NH);
    attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH);
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);   // determinism

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // Save input tensor as an image
    printf("Saving input tensor as image...\n");
    save_tensor_as_image(inp, B, T, C, "artifacts/in_cu.png");

    // Run the selected kernel
    printf("Running kernel\n");
    int block_size = 256; // Default block size
    forward1(d_out, d_preatt, d_att, d_inp, B, T, C, NH, block_size);

    // Copy the output tensor back to the host
    cudaCheck(cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));

    // Save output tensor as an image
    printf("Saving output tensor as image...\n");
    save_tensor_as_image(out, B, T, C, "artifacts/out_cu.png");

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));

    return 0;
}