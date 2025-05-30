#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <iomanip>

void serial_matrix_multiply(const std::vector<std::vector<double>>& A, 
                           const std::vector<std::vector<double>>& B, 
                           std::vector<std::vector<double>>& C, 
                           int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void parallel_matrix_multiply(const std::vector<std::vector<double>>& A, 
                             const std::vector<std::vector<double>>& B, 
                             std::vector<std::vector<double>>& C, 
                             int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    std::ofstream outfile("matrix_multiply_times.txt");
    outfile << "Size,SerialTime,ParallelTime\n";

    // 设置 OpenMP 线程数
    omp_set_num_threads(16); // 可根据硬件调整

    // 定义矩阵大小，从 128 到 4096
    std::vector<int> sizes = {128, 256, 384, 512, 
        640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 
        1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 
        2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096};

    for (int n : sizes) {
        // 初始化矩阵
        std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
        std::vector<std::vector<double>> B(n, std::vector<double>(n, 1.0));
        std::vector<std::vector<double>> C_serial(n, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> C_parallel(n, std::vector<double>(n, 0.0));

        // 串行执行
        auto start = std::chrono::high_resolution_clock::now();
        serial_matrix_multiply(A, B, C_serial, n);
        auto end = std::chrono::high_resolution_clock::now();
        // double serial_time = std::chrono::duration<double>(end - start).count(); // 单位：秒

        // 并行执行
        start = std::chrono::high_resolution_clock::now();
        parallel_matrix_multiply(A, B, C_parallel, n);
        end = std::chrono::high_resolution_clock::now();
        double parallel_time = std::chrono::duration<double>(end - start).count() *1000; // 单位：秒

        // 输出结果
        outfile << n <<    "," << parallel_time << "\n";
        std::cout << "Size: " << n << " , Parallel: " << parallel_time << "ms\n";
    }

    outfile.close();
    return 0;
}