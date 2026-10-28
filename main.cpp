// 这是第07课的回家作业，主题是访存优化
// 录播见：https://www.bilibili.com/video/BV1gu41117bW
// 作业的回答推荐写在 ANSWER.md 方便老师批阅，也可在 PR 描述中
// 请晒出程序被你优化前后的运行结果（ticktock 的用时统计）
// 可以比较采用了不同的优化手段后，加速了多少倍，做成表格
// 如能同时贴出 CPU 核心数量，缓存大小等就最好了（lscpu 命令）
// 作业中有很多个问句，请通过注释回答问题，并改进其代码，以使其更快
// 并行可以用 OpenMP 也可以用 TBB

#include <iostream>
//#include <x86intrin.h>  // _mm 系列指令都来自这个头文件
//#include <xmmintrin.h>  // 如果上面那个不行，试试这个
#include "ndarray.h"
#include "wangsrng.h"
#include "ticktock.h"
#include<algorithm>
// Matrix 是 YX 序的二维浮点数组：mat(x, y) = mat.data()[y * mat.shape(0) + x]
using Matrix = ndarray<2, float>;
// 注意：默认对齐到 64 字节，如需 4096 字节，请用 ndarray<2, float, AlignedAllocator<4096, float>>

static void matrix_randomize(Matrix &out) {
    TICK(matrix_randomize);
    size_t nx = out.shape(0);
    size_t ny = out.shape(1);

    // 这个循环为什么不够高效？如何优化？ 10 分
// #pragma omp parallel for collapse(2)
//for (int x = 0; x < nx; y++) {
// for (int y = 0; y < ny; y++) {
constexpr size_t blksz = 64;
#pragma omp parallel for 
    for (int ybase = 0; ybase < ny; ybase += blksz) {
        int minv = std::min(ny, ybase + blksz);
        for (int y = ybase; y < minv; ++y) {
            // for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float val = wangsrng(x, y).next_float();
                out(x, y) = val;
            }
        }
    }
    TOCK(matrix_randomize);
}

static void matrix_transpose(Matrix &out, Matrix const &in) {
    TICK(matrix_transpose);
    size_t nx = in.shape(0);
    size_t ny = in.shape(1);
    out.reshape(ny, nx);

    // 这个循环为什么不够高效？如何优化？ 15 分
    /*
    ndarray是列主序，原先的循环，对于out是缓存优化，对于in则不然
    疑问：分块循环后，为什么matrix_transpose执行时间稍微增加，反而调用它的matrix_RtAR的总用时减少了？
    */
    constexpr int  xblksz = 256;
    constexpr int  yblksz = 256;
#pragma omp parallel for collapse(2)
    // for (int x = 0; x < nx; x++) {
    //     for (int y = 0; y < ny; y++) {
    //         out(y, x) = in(x, y);
    //     }
    // }
    for (int xbase = 0; xbase < nx; xbase += xblksz) {
        for (int ybase = 0; ybase < ny; ybase += yblksz) {
            // for (int x = xbase; x - xbase < xblksz; ++x) {
            for (int x = xbase; (x - xbase < xblksz) && (x < nx); ++x) {
                // for (int y = 0; y - ybase < yblksz; y++) {
                for (int y = 0; (y - ybase < yblksz) && (y < ny); y++) {
                    out(y, x) = in(x, y);
                }
            }
        }
    }
    TOCK(matrix_transpose);
}

static void matrix_multiply(Matrix &out, Matrix const &lhs, Matrix const &rhs) {
    TICK(matrix_multiply);
    size_t nx = lhs.shape(0);
    size_t nt = lhs.shape(1);
    size_t ny = rhs.shape(1);
    if (rhs.shape(0) != nt) {
        std::cerr << "matrix_multiply: shape mismatch" << std::endl;
        throw;
    }
    out.reshape(nx, ny);

    // 这个循环为什么不够高效？如何优化？ 15 分
    /*sliede07，p157，寄存器分块，或者叫循环条块话*/
    constexpr int xblksz = 32;
 #pragma omp parallel for collapse(3)
    for (int y = 0; y < ny; y++) {
        for (int xbase = 0; xbase < nx; xbase += xblksz) {
        // for (int x = 0; x < nx; x++) {
            // out(x, y) = 0;  // 有没有必要手动初始化？ 5 分
            /*底层是vetor，数据类型是float，会初始化为0*/
            for (int t = 0; t < nt; t++) {
                for (int x = xbase; (x - xbase < xblksz) && (x < nx); ++x) {
                out(x, y) += lhs(x, t) * rhs(t, y);
                }
            }
        }
    }    
    TOCK(matrix_multiply);
}

// 求出 R^T A R
static void matrix_RtAR(Matrix &RtAR, Matrix const &R, Matrix const &A) {
    TICK(matrix_RtAR);
    // 这两个是临时变量，有什么可以优化的？ 5 分
    // Matrix Rt, RtA;  // 假设R(m, n)，A(p, q)，那么Rt(n, m),Rt可以与A相乘，那么A(m, q), RtA(n, q), RtAR(m, q);
    static Matrix Rt, RtA;  // 假设R(m, n)，A(p, q)，那么Rt(n, m),Rt可以与A相乘，那么A(m, q), RtA(n, q), RtAR(m, q);
    matrix_transpose(Rt, R);
    matrix_multiply(RtA, Rt, A);
    matrix_multiply(RtAR, RtA, R);
    TOCK(matrix_RtAR);
}

static float matrix_trace(Matrix const &in) {
    TICK(matrix_trace);
    float res = 0;
    size_t nt = std::min(in.shape(0), in.shape(1));
#pragma omp parallel for reduction(+:res)
    for (int t = 0; t < nt; t++) {
        res += in(t, t);
    }
    TOCK(matrix_trace);
    return res;
}

static void test_func(size_t n) {
    TICK(test_func);
    Matrix R(n, n);
    matrix_randomize(R);
    Matrix A(n, n);
    matrix_randomize(A);

    Matrix RtAR;
    matrix_RtAR(RtAR, R, A);

    std::cout << matrix_trace(RtAR) << std::endl;
    TOCK(test_func);
}

int main() {
    wangsrng rng;
    TICK(overall);
    for (int t = 0; t < 4; t++) {
        size_t n = 32 * (rng.next_uint64() % 16 + 24);
        std::cout << "t=" << t << ": n=" << n << std::endl;
        test_func(n);
    }
    TOCK(overall);
    return 0;
}