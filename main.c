#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"

// Размер тестовых матриц (NxN)
#define N 4

// 1. Тест Level 3 для float (Single Precision GEMM)
void test_sgemm() {
    printf("Running sgemm test\n", N, N);\

    float A[N*N] = { 1.0f,  2.0f,  3.0f,  4.0f, 
                     5.0f,  6.0f,  7.0f,  8.0f, 
                     1.0f,  1.0f,  1.0f,  1.0f, 
                     2.0f,  2.0f,  2.0f,  2.0f };
    
    float B[N*N] = { 1.0f,  1.0f,  1.0f,  1.0f, 
                     2.0f,  2.0f,  2.0f,  2.0f, 
                     3.0f,  3.0f,  3.0f,  3.0f, 
                     4.0f,  4.0f,  4.0f,  4.0f };

    float C[N*N] = { 0.0f };

    float expected[N*N] = { 30.0f, 30.0f, 30.0f, 30.0f, 
                            70.0f, 70.0f, 70.0f, 70.0f, 
                            10.0f, 10.0f, 10.0f, 10.0f, 
                            20.0f, 20.0f, 20.0f, 20.0f };

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);

    int passed = 1;
    for(int i = 0; i < N*N; i++) {
        if (fabsf(C[i] - expected[i]) > 1e-4) { 
            passed = 0;
            printf("Error at C[%d]: expected %f, got %f\n", i, expected[i], C[i]);
            break;
        }
    }

    if (passed) {
        printf("[PASS] cblas_sgemm (float matrices)\n");
    } else {
        printf("[FAIL] cblas_sgemm (float matrices)\n");
    }
}

// 2. Тест Level 3 для double (Double Precision GEMM)
void test_dgemm() {
    printf("Running dgemm test (%dx%d matrices)\n", N, N);
    
    double A[N*N] = { 1.0,  2.0,  3.0,  4.0, 
                      5.0,  6.0,  7.0,  8.0, 
                      1.0,  1.0,  1.0,  1.0, 
                      2.0,  2.0,  2.0,  2.0 };
    
    double B[N*N] = { 1.0,  1.0,  1.0,  1.0, 
                      2.0,  2.0,  2.0,  2.0, 
                      3.0,  3.0,  3.0,  3.0, 
                      4.0,  4.0,  4.0,  4.0 };
    
    double C[N*N] = { 0.0 };
    double expected[N*N] = { 30.0, 30.0, 30.0, 30.0, 
                             70.0, 70.0, 70.0, 70.0, 
                             10.0, 10.0, 10.0, 10.0, 
                             20.0, 20.0, 20.0, 20.0 };

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    int passed = 1;
    for(int i = 0; i < N*N; i++) {
        if (fabs(C[i] - expected[i]) > 1e-9) { 
            passed = 0;
            printf("Error at C[%d]: expected %f, got %f\n", i, expected[i], C[i]);
            break;
        }
    }

    if (passed) {
        printf("[PASS] cblas_dgemm (double matrices)\n");
    } else {
        printf("[FAIL] cblas_dgemm (double matrices)\n");
    }
}

// 3. Тест Level 3 для комплексных float (Complex Single GEMM)
void test_cgemm() {
    printf("Running cgemm test\n", N, N);

    float A[N*N*2] = { 0 }; 
    float B[N*N*2] = { 0 };
    float C[N*N*2] = { 0 };
    
    for(int i=0; i<N*N; i++) {
        A[i*2] = (float)(i + 1); 
        B[i*2] = 2.0f;
    }

    float alpha[2] = {1.0f, 0.0f}; 
    float beta[2] = {0.0f, 0.0f};  

    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                N, N, N, alpha, A, N, B, N, beta, C, N);

    if (fabsf(C[0] - 20.0f) < 1e-4 && fabsf(C[1] - 0.0f) < 1e-4) {
        printf("[PASS] cblas_cgemm (complex float matrices)\n");
    } else {
        printf("[FAIL] cblas_cgemm\n");
    }
}

// 4. Тест Level 3 для комплексных double (Complex Double GEMM)
void test_zgemm() {
    printf("Running zgemm test\n", N, N);
    
    double A[N*N*2] = { 0 }; 
    double B[N*N*2] = { 0 };
    double C[N*N*2] = { 0 };
    
    for(int i=0; i<N*N; i++) {
        A[i*2] = (double)(i + 1);
        B[i*2] = 2.0;
    }
    
    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                N, N, N, alpha, A, N, B, N, beta, C, N);

    if (fabs(C[0] - 20.0) < 1e-9 && fabs(C[1] - 0.0) < 1e-9) {
        printf("[PASS] cblas_zgemm (complex double matrices)\n");
    } else {
        printf("[FAIL] cblas_zgemm\n");
    }
}

int main() {
    printf("RUNNING CBLAS LEVEL 3 INTERFACE TESTS\n");
    
    test_sgemm();
    test_dgemm();
    test_cgemm();
    test_zgemm();
    
    return 0;
}