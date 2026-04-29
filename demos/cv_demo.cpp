/** @file cv_demo_cuda.cpp
 *
 * CUDA example - convolution on the image
 *
 * @author Dominik Belter (converted to CUDA)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#include "cuda_conv.cuh"

/// converts cv::Mat to vector
std::vector<unsigned char> toVector(const cv::Mat& image){
    return std::vector<unsigned char>(
        image.data,
        image.data + image.rows * image.cols
        );
}

/// converts vector to cv::Mat
void fromVector(const std::vector<unsigned char>& array, cv::Mat& image){
    if(array.size() == static_cast<size_t>(image.rows) * static_cast<size_t>(image.cols)){
        memcpy(image.data, array.data(), array.size());
    }
}

std::vector<unsigned char> computeConvCPU(const std::vector<unsigned char>& input,
                                          const std::vector<char>& mask,
                                          int rows,
                                          int cols)
{
    std::vector<unsigned char> output(input.size(), 0);

    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            int sum = 0;
            for(int dr = -1; dr <= 1; ++dr){
                for(int dc = -1; dc <= 1; ++dc){
                    int rr = r + dr;
                    int cc = c + dc;
                    if(rr >= 0 && rr < rows && cc >= 0 && cc < cols){
                        int mask_idx = (dr + 1) * 3 + (dc + 1);
                        sum += input[rr * cols + cc] * mask[mask_idx];
                    }
                }
            }
            if(sum < 0) sum = 0;
            else if(sum > 255) sum = 255;
            output[r * cols + c] = static_cast<unsigned char>(sum);
        }
    }

    return output;
}

int main()
{
    try {
        cv::Mat image = cv::imread("../../resources/messor2.jpg", cv::IMREAD_COLOR);

        if(image.empty()){
            std::cerr << "Warning: failed to load image from ../../resources/messor2.jpg. Using a synthetic test image instead." << std::endl;
            image = cv::Mat(256, 256, CV_8UC3);
            for(int r = 0; r < image.rows; ++r){
                for(int c = 0; c < image.cols; ++c){
                    image.at<cv::Vec3b>(r, c) = cv::Vec3b(static_cast<unsigned char>(c), static_cast<unsigned char>(r), 128);
                }
            }
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<unsigned char> input = toVector(gray);
        size_t vectorSize = input.size();
        std::vector<unsigned char> gpuOutput(vectorSize, 0);
        std::vector<unsigned char> cpuOutput(vectorSize, 0);

        std::vector<char> mask = {1, 1, 1, 0, 0, 0, -1, -1, -1};

        cv::imshow("Gray", gray);

        auto beginCPU = std::chrono::steady_clock::now();
        cpuOutput = computeConvCPU(input, mask, gray.rows, gray.cols);
        auto endCPU = std::chrono::steady_clock::now();

        auto beginGPU = std::chrono::steady_clock::now();
        computeConvCUDA(input.data(),
                        mask.data(),
                        gpuOutput.data(),
                        gray.rows,
                        gray.cols);
        auto endGPU = std::chrono::steady_clock::now();

        std::cout << "Time difference for CPU  = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(endCPU - beginCPU).count()
                  << "[µs]\n";
        std::cout << "Time difference for CUDA = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(endGPU - beginGPU).count()
                  << "[µs]\n";

        cv::Mat cpuResult = gray.clone();
        fromVector(cpuOutput, cpuResult);
        cv::imshow("Result CPU", cpuResult);

        cv::Mat gpuResult = gray.clone();
        fromVector(gpuOutput, gpuResult);
        cv::imshow("Result CUDA", gpuResult);

        cv::waitKey(0);
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
