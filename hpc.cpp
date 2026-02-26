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
cblas_dgem(Order,TransA,M,N,a,A,lda,x,inc x,b,y,inc y);

// C <-- aAB+bC
cblas_dgemm(Order,TransA,TransB,M,N,K,a,A,lda,B,ldb,b,C,ldc);



