#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "split_covar.h"
#include "momentum_sample.h"
#include "loop_vtx.h"
#include "vtx_interpol.h"

static int verify(double terr, double tol)
{
	int ret;
	if (terr > tol) {
		ret = 1;
		fprintf(stderr, "T-ERROR: %+.15E TOL: %+.0E TEST FAILED  ***\n\n", terr, tol);
	} else {
		ret = 0;
		fprintf(stderr, "T-ERROR: %+.15E TOL: %+.0E TEST PASSED\n\n", terr, tol);
	}

	return ret;
}

void test_lib_flow(void)
{
	unsigned long nke, nq;

	nke = 500;
	nq = 200;

	verify(test_zs_zsp_rot(nke, 35), 1E-7);

	verify(test_zs_split_covar(nke, nq, 1, 554), 1E-7);
	verify(test_zs_split_covar(nke, nq, -1, 5554), 1E-7);
	verify(test_zsp_split_covar(nke, nq, 1, 54), 1E-7);
	verify(test_zsp_split_covar(nke, nq, -1, 54), 1E-7);

	verify(test_interpolate_gma(10, nq, 1, 435), 1E-7);
	verify(test_interpolate_gma(10, nq, -1, 545), 1E-7);
}
