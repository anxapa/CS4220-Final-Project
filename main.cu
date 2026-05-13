// To compile:
// nvcc -o main main.cu -I $OPENCV_HOME/include/opencv4/ -L$OPENCV_HOME/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

#define RANDOM_SEED 42

// Check CUDA Errors
#define CHECK(call) {\
    const cudaError_t cuda_ret = call;\
    if (cuda_ret != cudaSuccess) {\
        printf("Error %s:%d", __FILE__, __LINE__);\
        printf("Code: %d, reason: %s", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1);\
    }\
}

#define CHANNELS 3

// Takes in current time of the system
// Usage: double <var> = myCPUTimer();
double myCPUTimer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    // timeval has two values: seconds (tv_sec) and microseconds (tv_usec)
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

// A CUDA Kernel that performs K-Means++ algorithm on an image to get weighted distances.
__global__ void kMeansPP_kernel(unsigned char* d_img, float* d_centroids, float* d_min_dists,
    int numPixels, int channels, int cIdx) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > numPixels) return;

    float dist = 0.0f;
    for (int ch = 0; ch < channels; ch++) {
        float diff = (float) d_img[i * channels + ch] - d_centroids[cIdx * channels + ch];
        dist += diff * diff;
    }

    if (cIdx == 0 || dist < d_min_dists[i]) {
        d_min_dists[i] = dist;
    }
}

// A CUDA Kernel that assigns each pixel to a centroid.
__global__ void assignClusters_kernel(unsigned char* d_img, float* d_centroids, int* d_labels,
    int numPixels, int channels, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numPixels) return;

    float minDist = FLT_MAX;
    int bestCluster = 0;

    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        // Finding distance
        for (int ch = 0; ch < channels; ch++) {
            float diff = (float) d_img[i * channels + ch] - d_centroids[c * channels + ch];
            dist += diff * diff;
        }
        // Finding cluster with least distance (best cluster)
        if (minDist > dist) {
            minDist = dist;
            bestCluster = c;
        }
    }

    d_labels[i] = bestCluster;
}

// A CUDA Kernel that maps the labels to the final quantized image
__global__ void applyQuantization_kernel(unsigned char* d_out_img, float* d_centroids, int* d_labels,
    int numPixels, int channels) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numPixels) return;

    int cluster = d_labels[i];
    for (int ch = 0; ch < channels; ch++) {
        d_out_img[i * channels + ch] = (unsigned char)d_centroids[cluster * channels + ch];
    }
}

// A CUDA Kernel that counts and sums up all the pixel values for each cluster.
__global__ void accumulate_clusters_kernel(unsigned char* d_img, int* d_labels, unsigned int* d_sums, int* d_counts,
    int numPixels, int channels) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numPixels) return;

    int cluster = d_labels[i];

    // Atomic increments to avoid race conditions
    atomicAdd(&d_counts[cluster], 1);
    for (int ch = 0; ch < channels; ch++) {
        atomicAdd(&d_sums[cluster * channels + ch], (unsigned int) d_img[i * channels + ch]);
    }
}

// A host function that handles all the device memory allocation / copying, etc.
// for device quantization.
void quantizeImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int k, double* times) {
    double startTime, endTime;

    // Image variables
    int width = Pin_Mat_h.cols;
    int height = Pin_Mat_h.rows;
    int channels = Pin_Mat_h.channels();
    int numPixels = width * height;
    size_t imgSize = numPixels * channels * sizeof(unsigned char);

    unsigned char* h_img = Pin_Mat_h.data;

    // Device variables
    unsigned char *d_img, *d_out_img;
    float *d_centroids, *d_min_dists;
    int *d_labels, *d_counts;
    unsigned int *d_sums;

    CHECK(cudaMalloc(&d_img, imgSize));
    CHECK(cudaMalloc(&d_out_img, imgSize));
    CHECK(cudaMalloc(&d_centroids, k * channels * sizeof(float)));
    CHECK(cudaMalloc(&d_min_dists, numPixels * sizeof(float)));
    CHECK(cudaMalloc(&d_labels, numPixels * sizeof(int)));
    CHECK(cudaMalloc(&d_sums, k * channels * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_counts, k * sizeof(int)));

    CHECK(cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (numPixels + blockSize - 1) / blockSize;

    /*
    =========================================================
    Stage 1: Obtaining the initial centroids through K-Means++
    =========================================================
    */
    startTime = myCPUTimer();
    float* h_centroids = new float[k * channels];
    float* h_min_dists = new float[numPixels];

    // Find first centroid
    int firstIdx = rand() % numPixels;
    for (int ch = 0; ch < channels; ch++) {
        h_centroids[ch] = (float)h_img[firstIdx * channels + ch];
    }
    CHECK(cudaMemcpy(d_centroids, h_centroids, channels * sizeof(float), cudaMemcpyHostToDevice));

    // Find other centroids
    for (int c = 1; c < k; c++) {
        kMeansPP_kernel<<<numBlocks, blockSize>>>(d_img, d_centroids, d_min_dists, numPixels, channels, c - 1);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_min_dists, d_min_dists, numPixels * sizeof(float), cudaMemcpyDeviceToHost));

        // Weighted random
        double totalWeight = 0.0;
        for (int i = 0; i < numPixels; i++) totalWeight += h_min_dists[i];

        double r = ((double) rand() / RAND_MAX) * totalWeight;
        double cumulative = 0.0;
        int chosenIdx = numPixels - 1;

        for (int i = 0; i < numPixels; i++) {
            cumulative += h_min_dists[i];
            if (cumulative >= r) {
                chosenIdx = i;
                break;
            }
        }

        // Assign new centroid
        for (int ch = 0; ch < channels; ch++) {
            h_centroids[c * channels + ch] = (float)h_img[chosenIdx * channels + ch];
        }

        CHECK(cudaMemcpy(d_centroids, h_centroids, k * channels * sizeof(float), cudaMemcpyHostToDevice));
    }
    endTime = myCPUTimer();
    times[0] = endTime - startTime;

    /*
    =========================================================
    Stage 2: Calculating the final centroids through K-Means
    =========================================================
    */
    startTime = myCPUTimer();
    bool changed = true;
    int max_iters = 100;
    int iter = 0;

    unsigned int* h_sums = new unsigned int[k * channels];
    int* h_counts = new int[k];

    while (changed && iter < max_iters) {
        changed = false;

        CHECK(cudaMemset(d_sums, 0, k * channels * sizeof(unsigned int)));
        CHECK(cudaMemset(d_counts, 0, k * sizeof(int)));

        assignClusters_kernel<<<numBlocks, blockSize>>>(d_img, d_centroids, d_labels, numPixels, channels, k);
        CHECK(cudaDeviceSynchronize());

        accumulate_clusters_kernel<<<numBlocks, blockSize>>>(d_img, d_labels, d_sums, d_counts, numPixels, channels);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_sums, d_sums, k * channels * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_counts, d_counts, k * sizeof(int), cudaMemcpyDeviceToHost));
    
        for (int c = 0; c < k; c++) {
            if (h_counts[c] == 0) continue;

            for(int ch = 0; ch < channels; ch++) {
                float newVal = floor((float)h_sums[c * channels + ch] / h_counts[c]);
                if (h_centroids[c * channels + ch] != newVal) {
                    changed = true;
                    h_centroids[c * channels + ch] = newVal;
                }
            }
        }

        CHECK(cudaMemcpy(d_centroids, h_centroids, k * channels * sizeof(float), cudaMemcpyHostToDevice));
        iter++;
    }
    endTime = myCPUTimer();
    times[1] = endTime - startTime;

    /*
    =========================================================
    Stage 3: Reassigning new colors (centroids) to the output
    =========================================================
    */
    startTime = myCPUTimer();
    applyQuantization_kernel<<<numBlocks, blockSize>>>(d_out_img, d_centroids, d_labels, numPixels, channels);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(Pout_Mat_h.data, d_out_img, imgSize, cudaMemcpyDeviceToHost));
    endTime = myCPUTimer();
    times[2] = endTime - startTime;
    
    delete[] h_centroids;
    delete[] h_min_dists;
    delete[] h_counts;
    delete[] h_sums;
    cudaFree(d_img);
    cudaFree(d_out_img);
    cudaFree(d_centroids);
    cudaFree(d_min_dists);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);
}

// A host function that runs quantization within the host.
void quantizeImage_h(cv::Mat output, cv::Mat input, int k, double* times) {
    int rows = input.rows;
    int cols = input.cols;
    int num_pixels = rows * cols;
    double startTime, endTime;

    // Use float for centroid calculations to maintain precision
    std::vector<cv::Vec3f> centroids(k);
    std::vector<int> labels(num_pixels);

    /*
    =========================================================
    Stage 1: Obtaining the initial centroids through K-Means++
    =========================================================
    */

    startTime = myCPUTimer();

    // Pick the first centroid randomly from the pixel data
    int first_idx = rand() % num_pixels;
    centroids[0] = input.at<cv::Vec3b>(first_idx / cols, first_idx % cols);

    std::vector<float> min_dists(num_pixels, FLT_MAX);
    
    for (int c = 1; c < k; c++) {
        double total_dist = 0;
        
        // Calculate distance to the nearest existing centroid for every pixel
        for (int i = 0; i < num_pixels; i++) {
            cv::Vec3f pix = input.at<cv::Vec3b>(i / cols, i % cols);
            // NORM_L2SQR => Squared Euclidean Distance
            float d = cv::norm(pix - centroids[c - 1], cv::NORM_L2SQR);
            if (d < min_dists[i]) min_dists[i] = d;
            total_dist += min_dists[i];
        }

        // Weighted random selection for the next centroid
        double r = ((double)rand() / RAND_MAX) * total_dist;
        double accum = 0;
        for (int i = 0; i < num_pixels; i++) {
            accum += min_dists[i];
            if (accum >= r) {
                centroids[c] = input.at<cv::Vec3b>(i / cols, i % cols);
                break;
            }
        }
    }

    endTime = myCPUTimer();
    times[0] = endTime - startTime;

    /*
    =========================================================
    Stage 2: Calculating the final centroids through K-Means
    =========================================================
    */
    startTime = myCPUTimer();
    bool changed = true;
    int max_iters = 100;
    int iter = 0;
    
    while (changed && iter < max_iters) {
        changed = false;
        std::vector<cv::Vec3f> new_sums(k, cv::Vec3f(0, 0, 0));
        std::vector<int> counts(k, 0);

        // Assignment Step: Each pixel finds its closest centroid
        for (int i = 0; i < num_pixels; i++) {
            cv::Vec3f pix = (cv::Vec3f)input.at<cv::Vec3b>(i / cols, i % cols);
            float best_d = FLT_MAX;
            int best_c = 0;
            
            for (int c = 0; c < k; c++) {
                float d = cv::norm(pix - centroids[c], cv::NORM_L2SQR);
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            labels[i] = best_c;
        }

        // Update Step: Accumulate the colors for the new centroids
        for (int i = 0; i < num_pixels; i++) {
            new_sums[labels[i]] += (cv::Vec3f)input.at<cv::Vec3b>(i / cols, i % cols);
            counts[labels[i]]++;
        }

        // Finalize new centroids
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                cv::Vec3f next_c = new_sums[c] / (float)counts[c];
                next_c[0] = floor(next_c[0]);
                next_c[1] = floor(next_c[1]);
                next_c[2] = floor(next_c[2]);
                
                if (centroids[c] != next_c) {
                    centroids[c] = next_c;
                    changed = true;
                }
            }
        }
        iter++;
    }
    endTime = myCPUTimer();
    times[1] = endTime - startTime;

    /*
    =========================================================
    Stage 3: Reassigning new colors (centroids) to the output
    =========================================================
    */
    startTime = myCPUTimer();
    for (int i = 0; i < num_pixels; i++) {
        output.at<cv::Vec3b>(i / cols, i % cols) = (cv::Vec3b)centroids[labels[i]];
    }
    endTime = myCPUTimer();
    times[2] = endTime - startTime;
}

// Helper to get the mean of a vector of doubles.
double getMean(const std::vector<double>& times) {
    if (times.empty()) return 0.0;
    double sum = 0.0;
    for (double t : times) sum += t;
    return sum / times.size();
}

// Helper function to calculate and print general benchmark statistics.
void printBenchmarkStats(const char* name, const std::vector<double>& times) {
    if (times.empty()) return;

    double sum = 0.0, min_time = times[0], max_time = times[0];
    for (double t : times) {
        sum += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    double mean = sum / times.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    double stddev = sqrt(variance / times.size());

    printf("%-25s | %8.4f s | %8.4f s | %8.4f s | %8.4f s\n", name, mean, stddev, min_time, max_time);
}

// The core benchmark function.
void runBenchmark(cv::Mat& origImg, unsigned int nColors, int iterations) {
    int height = origImg.rows; 
    int width = origImg.cols;
    
    // Total times
    std::vector<double> cpu_times;
    std::vector<double> gpu_times;
    
    // Breakdown times for CPU and GPU
    std::vector<double> c_s1, c_s2, c_s3;
    std::vector<double> g_s1, g_s2, g_s3;

    cv::Mat quantizedImg_cpu(height, width, CV_8UC3, cv::Scalar(0));
    cv::Mat quantizedImg_gpu(height, width, CV_8UC3, cv::Scalar(0));

    printf("================================================================================\n");
    printf("Running Benchmark: %d Iterations, %d Colors, Image Size: %dx%d\n", iterations, nColors, width, height);
    printf("================================================================================\n");

    // Warm-up run for GPU to initialize CUDA context (prevents skewed first-run metrics)
    printf("Warming up GPU...\n");
    double dummy_times[3];
    quantizeImage_d(quantizedImg_gpu, origImg, nColors, dummy_times);

    printf("Executing benchmark runs...\n");
    for (int i = 0; i < iterations; i++) {
        double stage_times[3];

        // Reset seed to ensure identical work/initialization per iteration 
        srand(RANDOM_SEED);

        // Benchmark CPU
        double startTime = myCPUTimer();
        quantizeImage_h(quantizedImg_cpu, origImg, nColors, stage_times);
        double endTime = myCPUTimer();
        
        cpu_times.push_back(endTime - startTime);
        c_s1.push_back(stage_times[0]);
        c_s2.push_back(stage_times[1]);
        c_s3.push_back(stage_times[2]);

        // Reset seed again for fair GPU comparison
        srand(RANDOM_SEED);

        // Benchmark GPU
        startTime = myCPUTimer();
        quantizeImage_d(quantizedImg_gpu, origImg, nColors, stage_times);
        endTime = myCPUTimer();
        
        gpu_times.push_back(endTime - startTime);
        g_s1.push_back(stage_times[0]);
        g_s2.push_back(stage_times[1]);
        g_s3.push_back(stage_times[2]);
        
        printf("\rProgress: [%d/%d] iterations complete.", i + 1, iterations);
        fflush(stdout);
    }
    printf("\n\n");

    // Output Overall Results Table
    printf("================================================================================\n");
    printf("Overall Execution Metrics\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-25s | %-10s | %-10s | %-10s | %-10s\n", "Method", "Mean", "StdDev", "Min", "Max");
    printf("--------------------------------------------------------------------------------\n");
    printBenchmarkStats("CPU Sequential", cpu_times);
    printBenchmarkStats("GPU CUDA", gpu_times);
    
    // Output Stage Breakdown Table
    printf("\n================================================================================\n");
    printf("Stage-by-Stage Breakdown (Mean Times)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-25s | %-14s | %-14s | %-14s\n", "Method", "Stage 1 (Init)", "Stage 2 (Iter)", "Stage 3 (Map)");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-25s | %12.4f s | %12.4f s | %12.4f s\n", "CPU Sequential", getMean(c_s1), getMean(c_s2), getMean(c_s3));
    printf("%-25s | %12.4f s | %12.4f s | %12.4f s\n", "GPU CUDA", getMean(g_s1), getMean(g_s2), getMean(g_s3));
    printf("================================================================================\n");

    // Calculate overall speedup based on mean times
    double cpu_mean = getMean(cpu_times);
    double gpu_mean = getMean(gpu_times);

    printf("\nAverage Overall Speedup (CPU / GPU): %.2fX\n\n", cpu_mean / gpu_mean);

    // Save a copy of quantized images to disk
    bool check = cv::imwrite("./quantizedImg_cpu.png", quantizedImg_cpu);
    if (check == false){ printf("error!\n"); return;}
    check = cv::imwrite("./quantizedImg_gpu.png", quantizedImg_gpu);
    if (check == false){ printf("error!\n"); return;}
}

// Function that runs the quantization once for both CPU and GPU and prints a summary.
void runSinglePass(cv::Mat& origImg, unsigned int nColors) {
    int height = origImg.rows;
    int width = origImg.cols;

    cv::Mat quantizedImg_cpu(height, width, CV_8UC3, cv::Scalar(0));
    cv::Mat quantizedImg_gpu(height, width, CV_8UC3, cv::Scalar(0));

    double cpu_stages[3] = {0}, gpu_stages[3] = {0};

    printf("========================================================\n");
    printf("Single Pass Execution: %d Colors, Image: %dx%d\n", nColors, width, height);
    printf("========================================================\n");

    // --- CPU Execution ---
    srand(RANDOM_SEED);
    double cpu_start = myCPUTimer();
    quantizeImage_h(quantizedImg_cpu, origImg, nColors, cpu_stages);
    double cpu_total = myCPUTimer() - cpu_start;

    // --- GPU Execution ---
    // Note: Warm-up is skipped here for a "raw" single-run feel, 
    // but the stage timers will still be accurate.
    srand(RANDOM_SEED);
    double gpu_start = myCPUTimer();
    quantizeImage_d(quantizedImg_gpu, origImg, nColors, gpu_stages);
    double gpu_total = myCPUTimer() - gpu_start;

    // --- Summary Output ---
    printf("%-20s | %-10s | %-10s\n", "Metric", "CPU (s)", "GPU (s)");
    printf("--------------------------------------------------------\n");
    printf("%-20s | %10.4f | %10.4f\n", "Stage 1 (Init)", cpu_stages[0], gpu_stages[0]);
    printf("%-20s | %10.4f | %10.4f\n", "Stage 2 (K-Means)", cpu_stages[1], gpu_stages[1]);
    printf("%-20s | %10.4f | %10.4f\n", "Stage 3 (Mapping)", cpu_stages[2], gpu_stages[2]);
    printf("--------------------------------------------------------\n");
    printf("%-20s | %10.4f | %10.4f\n", "Total Time", cpu_total, gpu_total);
    printf("========================================================\n");
    printf("Speedup: %.2fX\n\n", cpu_total / gpu_total);

    // Save results
    cv::imwrite("./quantizedImg_cpu.png", quantizedImg_cpu);
    cv::imwrite("./quantizedImg_gpu.png", quantizedImg_gpu);
    printf("Results saved to quantizedImg_cpu.png and quantizedImg_gpu.png\n");
}

int main(int argc, char** argv) {
    // Input validation
    if (argc < 3 || argc > 4) {
        printf("Usage: ./main <image_path> <#_of_colors> [iterations]\n");
        printf("  - No [iterations]: Runs a single-pass summary.\n");
        printf("  - With [iterations]: Runs full benchmark suite.\n");
        return -1;
    }

    // Variables
    unsigned int nColors = atoi(argv[2]);
    
    // Use OpenCV to load image in BGR channel
    cv::Mat origImg = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(origImg.empty()) {
        printf("Error: image not found at path '%s'!\n", argv[1]); 
        return -1;
    }

    if (argc == 3) {
        runSinglePass(origImg, nColors);
    } else {
        int iterations = atoi(argv[3]);
        runBenchmark(origImg, nColors, iterations);
    }

    return 0;
}