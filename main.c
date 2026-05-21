#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"

// 1. Тест для float (Single precision)
void test_saxpy() {
    float alpha = 2.0f;
    float X[2] = {1.0f, 2.0f};
    float Y[2] = {3.0f, 4.0f};
    float expected[2] = {5.0f, 8.0f}; 

    cblas_saxpy(2, alpha, X, 1, Y, 1);

    if (fabsf(Y[0] - expected[0]) < 1e-5 && fabsf(Y[1] - expected[1]) < 1e-5) {
        printf("[PASS] cblas_saxpy (float)\n");
    } else {
        printf("[FAIL] cblas_saxpy (float)\n");
    }
}

// 2. Тест для double (Double precision)
void test_daxpy() {
    double alpha = 3.0;
    double X[2] = {1.0, 2.0};
    double Y[2] = {1.0, 1.0};
    double expected[2] = {4.0, 7.0}; // 3*1+1, 3*2+1

    cblas_daxpy(2, alpha, X, 1, Y, 1);

    if (fabs(Y[0] - expected[0]) < 1e-9 && fabs(Y[1] - expected[1]) < 1e-9) {
        printf("[PASS] cblas_daxpy (double)\n");
    } else {
        printf("[FAIL] cblas_daxpy (double)\n");
    }
}

// 3. Тест для комплексных float (Complex Single)
void test_caxpy() {
    float alpha[2] = {2.0f, 0.0f}; 
    float X[4] = {1.0f, 1.0f, 2.0f, 2.0f}; 
    float Y[4] = {1.0f, 1.0f, 1.0f, 1.0f}; 
    
    cblas_caxpy(2, alpha, X, 1, Y, 1);

    if (fabsf(Y[0] - 3.0f) < 1e-5 && fabsf(Y[1] - 3.0f) < 1e-5) {
        printf("[PASS] cblas_caxpy (complex float)\n");
    } else {
        printf("[FAIL] cblas_caxpy (complex float)\n");
    }
}

// 4. Тест для комплексных double (Complex Double)
void test_zaxpy() {
    double alpha[2] = {2.0, 0.0}; 
    double X[4] = {1.0, 1.0, 2.0, 2.0};
    double Y[4] = {1.0, 1.0, 1.0, 1.0};
    
    cblas_zaxpy(2, alpha, X, 1, Y, 1);

    if (fabs(Y[0] - 3.0) < 1e-9 && fabs(Y[1] - 3.0) < 1e-9) {
        printf("[PASS] cblas_zaxpy (complex double)\n");
    } else {
        printf("[FAIL] cblas_zaxpy (complex double)\n");
    }
}

int main() {
    printf("RUNNING CBLAS INTERFACE TESTS\n");
    
    test_saxpy();
    test_daxpy();
    test_caxpy();
    test_zaxpy();
    
    printf("=====================================\n");
    return 0;
}