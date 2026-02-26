#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cblas.h"
using namespace std;

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


