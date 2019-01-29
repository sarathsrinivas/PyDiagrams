#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include <lib_gpr/lib_gpr.h>
#include <lib_rng/lib_rng.h>
#include "lib_flow.h"

#define PI (3.1415926535897)

void get_I23(double *I23, const double *qmin, const double *qmax, unsigned long nth, double lq)
{
	double q0, q1, q02, q12, q03, q13, q04, q14, q05, q15, q06, q16, l, l2, l3, l4, expq, erfq, expqi, t1,
	    t2, t21, t22, sqpi;
	unsigned long i;

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;
	sqpi = sqrt(PI);

	for (i = 0; i < nth; i++) {

		q0 = qmin[i];
		q1 = qmax[i];

		q02 = q0 * q0;
		q12 = q1 * q1;
		q03 = q02 * q0;
		q13 = q12 * q1;
		q04 = q03 * q0;
		q14 = q13 * q1;
		q05 = q04 * q0;
		q15 = q14 * q1;
		q06 = q05 * q0;
		q16 = q15 * q1;

		expq = exp(-(q0 - q1) * (q0 - q1) / l2);
		expqi = 1.0 / expq;
		erfq = Erf((q0 - q1) / l);

		t1 = 8 * l * (q03 + q13) * (l2 + q02 + q0 * q1 + q12);

		t21 = sqpi * (3 * l2 * (q14 - q04) - 2 * q06 + 2 * q16) * erfq;

		t22 = 2 * l * (l2 * (q03 + q13) + 3 * (q05 + q15));

		t2 = -4 * expqi * (t21 + t22);

		I23[i] = (1.0 / 48.0) * l * expq * (t1 + t2);
	}
}

void get_I22(double *I22, const double *qmin, const double *qmax, unsigned long nth, double lq)
{
	double q0, q1, q02, q12, q03, q13, q04, q14, q05, q15, q06, q16, l, l2, l3, l4, expq, erfq, expqi, t1,
	    t11, t12, t2, t21, t22, sqpi;
	unsigned long i;

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;
	sqpi = sqrt(PI);

	for (i = 0; i < nth; i++) {

		q0 = qmin[i];
		q1 = qmax[i];

		q02 = q0 * q0;
		q12 = q1 * q1;
		q03 = q02 * q0;
		q13 = q12 * q1;
		q04 = q03 * q0;
		q14 = q13 * q1;
		q05 = q04 * q0;
		q15 = q14 * q1;
		q06 = q05 * q0;
		q16 = q15 * q1;

		expq = exp(-(q0 - q1) * (q0 - q1) / l2);
		expqi = 1.0 / expq;
		erfq = Erf((q0 - q1) / l);

		t11 = 3 * (q04 + q03 * q1 + q02 * q12 + q0 * q13 + q14);

		t1 = 2 * l * (l4 + l2 * (q0 - q1) * (q0 - q1) + t11);

		t21 = sqpi * (5 * l2 * (q13 - q03) - 6 * q05 + 6 * q15) * erfq;

		t22 = l * (2 * l4 + 15 * (q04 + q14));

		t2 = -1 * expqi * (t21 + t22);

		I22[i] = (1.0 / 30.0) * l * expq * (t1 + t2);
	}
}

void get_I33(double *I33, const double *qmin, const double *qmax, unsigned long nth, double lq)
{
	double q0, q1, q02, q12, q03, q13, q04, q14, q05, q15, q06, q16, q07, q17, l, l2, l3, l4, l5, l6,
	    expq, erfq, expqi, t1, t11, t12, t2, t21, t22, t23, sqpi;
	unsigned long i;

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;
	l5 = l4 * l;
	l6 = l5 * l;
	sqpi = sqrt(PI);

	for (i = 0; i < nth; i++) {

		q0 = qmin[i];
		q1 = qmax[i];

		q02 = q0 * q0;
		q12 = q1 * q1;
		q03 = q02 * q0;
		q13 = q12 * q1;
		q04 = q03 * q0;
		q14 = q13 * q1;
		q05 = q04 * q0;
		q15 = q14 * q1;
		q06 = q05 * q0;
		q16 = q15 * q1;
		q07 = q06 * q0;
		q17 = q16 * q1;

		expq = exp(-(q0 - q1) * (q0 - q1) / l2);
		expqi = 1.0 / expq;
		erfq = Erf((q0 - q1) / l);

		t11 = 2 * sqpi * (21 * l2 * (q05 - q15) + 10 * (q07 - q17)) * erfq;

		t12 = l * (6 * l6 - 35 * l2 * (q04 + q14) - 70 * (q06 + q16));

		t1 = expqi * (t11 + t12);

		t21 = -3 * l6 - 3 * l4 * (q0 - q1) * (q0 - q1);

		t22 = l2 * (16 * q04 + 6 * q03 * q1 - 9 * q02 * q12 + 6 * q0 * q13 + 16 * q14);

		t23 = 10 * (q06 + q05 * q1 + q04 * q12 + q03 * q13 + q02 * q14 + q0 * q15 + q16);

		t2 = 2 * l * (t21 + t22 + t23);

		I33[i] = (1.0 / 140.0) * l * expq * (t1 + t2);
	}
}

double test_Imn(double qmin, double qmax, unsigned long nq)
{
	double I22[1], I22_num, I33[1], I33_num, I23[1], I23_num, *q, *wq, rel_err, qi2, qi3, q2, q3, Kqqi,
	    lq, q0[1], q1[1];
	unsigned long i, j;

	q = malloc(nq * sizeof(double));
	assert(q);
	wq = malloc(nq * sizeof(double));
	assert(wq);

	q0[0] = qmin;
	q1[0] = qmax;

	gauss_grid_create(nq, q, wq, qmin, qmax);

	lq = 1.0;

	I22_num = 0;
	I23_num = 0;
	I33_num = 0;
	for (i = 0; i < nq; i++) {
		for (j = 0; j < nq; j++) {

			q2 = q[j] * q[j];
			q3 = q2 * q[j];
			qi2 = q[i] * q[i];
			qi3 = qi2 * q[i];

			Kqqi = exp(-(q[i] - q[j]) * (q[i] - q[j]) / (lq * lq));

			I22_num += qi2 * q2 * Kqqi * wq[i] * wq[j];
			I23_num += qi2 * q3 * Kqqi * wq[i] * wq[j];
			I33_num += qi3 * q3 * Kqqi * wq[i] * wq[j];
		}
	}

	get_I22(I22, q0, q1, 1, lq);
	get_I23(I23, q0, q1, 1, lq);
	get_I33(I33, q0, q1, 1, lq);

	rel_err = 0;
	rel_err += fabs(I22_num - I22[0]) / fabs(I22_num + I22[0]);
	rel_err += fabs(I23_num - I23[0]) / fabs(I23_num + I23[0]);
	rel_err += fabs(I33_num - I33[0]) / fabs(I33_num + I33[0]);

	printf("%+.15E %+.15E\n", I22[0], I22_num);
	printf("%+.15E %+.15E\n", I23[0], I23_num);
	printf("%+.15E %+.15E\n", I33[0], I33_num);

	free(q);
	free(wq);

	return rel_err;
}
