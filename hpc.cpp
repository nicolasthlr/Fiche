g++ main.cpp -llapack -lblas


#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cblas.h"
using namespace std;

// Cblas (replace "d" by "z" if we deal with complex)

// y <-- ax+y
cblas_daxpy(n,a,x,inc x,y,inc y);

// y <-- x
cblas_dcopy(n,x,inc x,y,inc y);

// x . y
cblas_ddot(n,x,inc x, y, inc y);

// Euclidian norm || x ||2
cblas_dnrm2(n,x,inc x);

// x <-- ax
cblas_dscal(n,a,x,inc x);

// y <-- aAx + by
// Order = CblasRowMajor or CblasColMajor, TransA = CblasNoTrans or CblasTrans, lda = nb colonnes (en row major) 
cblas_dgemv(Order,TransA,M,N,a,A,lda,x,inc x,b,y,inc y);

// C <-- aAB+bC
cblas_dgemm(Order,TransA,TransB,M,N,K,a,A,lda,B,ldb,b,C,ldc);


#define F77NAME(x) x##_

// Solve Ax = b
// On input b contains RHS vector, on output b contains solution
extern "C" {
  void F77NAME(dgesv)(const int& n, const int& nrhs, const double * A,
                      const int& lda, int * ipiv, double * b,
                      const int& ldb, int& info);
}

// Eigenvalues/eigenvectors of symmetric matrix
extern "C" {
  void F77NAME(dsyev)(const char& v, const char& ul, const int& n,
                      double* a, const int& lda, double* w,
                      double* work, const int& lwork, int* info);
}

// Decomposition LU 
extern "C" {
  void F77NAME(dgetrf)(const int& m, const int& n, double* a, 
                       const int& lda, int* ipiv, int& info);
}


// Example
int n = 50; // Problem size
int nrhs = 1; // Number of RHS vectors
int info = 0;
double* A = new double[n*n];
int* ipiv = new int[n]; // Vector for pivots
double* b = new double[n]; // RHS vector / output vector




