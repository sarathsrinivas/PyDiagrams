#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include <lib_gpr/lib_gpr.h>
#include <lib_rng/lib_rng.h>
#include "lib_flow.h"

#define PI (3.1415926535897)

double get_I22(double q0, double q1, double qi0, double qi1, double lq)
{
	double I22, q02, q03, q04, q05, q12, q13, q14, q15, l, l2, l3, l4, l5, l6, qi02, qi03, qi04, qi05,
	    qi12, qi13, qi14, qi15, sqpi, qdiff00, qdiff01, qdiff10, qdiff11;

	sqpi = sqrt(PI);

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;

	q02 = q0 * q0;
	q03 = q02 * q0;
	q04 = q03 * q0;
	q05 = q04 * q0;

	q12 = q1 * q1;
	q13 = q12 * q1;
	q14 = q13 * q1;
	q15 = q14 * q1;

	qi02 = qi0 * qi0;
	qi03 = qi02 * qi0;
	qi04 = qi03 * qi0;
	qi05 = qi04 * qi0;

	qi12 = qi1 * qi1;
	qi13 = qi12 * qi1;
	qi14 = qi13 * qi1;
	qi15 = qi14 * qi1;

	qdiff00 = (q0 - qi0) * (q0 - qi0);
	qdiff01 = (q0 - qi1) * (q0 - qi1);
	qdiff10 = (q1 - qi0) * (q1 - qi0);
	qdiff11 = (q1 - qi1) * (q1 - qi1);

	I22 = (l
	       * (2 * l
		      * ((-l4 - l2 * qdiff00 - 3 * (q04 + q03 * qi0 + q02 * qi02 + q0 * qi03 + qi04))
			     * exp(-qdiff00 / l2)
			 + (l4 + l2 * qdiff10 + 3 * (q14 + q13 * qi0 + q12 * qi02 + q1 * qi03 + qi04))
			       * exp(-qdiff10 / l2)
			 + (l4 + l2 * qdiff01 + 3 * (q04 + q03 * qi1 + q02 * qi12 + q0 * qi13 + qi14))
			       * exp(-qdiff01 / l2)
			 + (-l4 - l2 * qdiff11 - 3 * (q14 + q13 * qi1 + q12 * qi12 + q1 * qi13 + qi14))
			       * exp(-qdiff11 / l2))
		  + sqpi * (-6 * q05 + 6 * qi05 + 5 * l2 * (-q03 + qi03)) * Erf((q0 - qi0) / l)
		  + sqpi * (5 * l2 * (q13 - qi03) + 6 * (q15 - qi05)) * Erf((q1 - qi0) / l)
		  + sqpi
			* ((5 * l2 * (q03 - qi13) + 6 * (q05 - qi15)) * Erf((q0 - qi1) / l)
			   + qi13 * (5 * l2 + 6 * qi12) * Erf((q1 - qi1) / l)
			   + q13 * (5 * l2 + 6 * q12) * Erf((-q1 + qi1) / l))))
	      / 60.0;

	return I22;
}

double get_I23(double q0, double q1, double qi0, double qi1, double lq)
{
	double I23, q02, q03, q04, q05, q06, q12, q13, q14, q15, q16, l, l2, l4, l3, l5, l6, qi02, qi03, qi04,
	    qi05, qi06, qi12, qi13, qi14, qi15, qi16, sqpi, qdiff00, qdiff01, qdiff10, qdiff11;

	sqpi = sqrt(PI);

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;
	l5 = l4 * l;
	l6 = l5 * l;

	q02 = q0 * q0;
	q03 = q02 * q0;
	q04 = q03 * q0;
	q05 = q04 * q0;
	q06 = q05 * q0;

	q12 = q1 * q1;
	q13 = q12 * q1;
	q14 = q13 * q1;
	q15 = q14 * q1;
	q16 = q15 * q1;

	qi02 = qi0 * qi0;
	qi03 = qi02 * qi0;
	qi04 = qi03 * qi0;
	qi05 = qi04 * qi0;
	qi06 = qi05 * qi0;

	qi12 = qi1 * qi1;
	qi13 = qi12 * qi1;
	qi14 = qi13 * qi1;
	qi15 = qi14 * qi1;
	qi16 = qi15 * qi1;

	qdiff00 = (q0 - qi0) * (q0 - qi0);
	qdiff01 = (q0 - qi1) * (q0 - qi1);
	qdiff10 = (q1 - qi0) * (q1 - qi0);
	qdiff11 = (q1 - qi1) * (q1 - qi1);

	I23 = (l
	       * ((-2 * l
		   * (3 * l4 * (q0 - qi0) + 2 * l2 * (q03 - 3 * q02 * qi0 + 3 * q0 * qi02 + 7 * qi03)
		      + 8 * (q05 + q04 * qi0 + q03 * qi02 + q02 * qi03 + q0 * qi04 + qi05)))
		      * exp(-qdiff00 / l2)
		  + (2 * l
		     * (3 * l4 * (q1 - qi0) + 2 * l2 * (q13 - 3 * q12 * qi0 + 3 * q1 * qi02 + 7 * qi03)
			+ 8 * (q15 + q14 * qi0 + q13 * qi02 + q12 * qi03 + q1 * qi04 + qi05)))
			* exp(-qdiff10 / l2)
		  + (2 * l
		     * (3 * l4 * (q0 - qi1) + 2 * l2 * (q03 - 3 * q02 * qi1 + 3 * q0 * qi12 + 7 * qi13)
			+ 8 * (q05 + q04 * qi1 + q03 * qi12 + q02 * qi13 + q0 * qi14 + qi15)))
			* exp(-qdiff01 / l2)
		  - (2 * l
		     * (3 * l4 * (q1 - qi1) + 2 * l2 * (q13 - 3 * q12 * qi1 + 3 * q1 * qi12 + 7 * qi13)
			+ 8 * (q15 + q14 * qi1 + q13 * qi12 + q12 * qi13 + q1 * qi14 + qi15)))
			* exp(-qdiff11 / l2)
		  + sqpi * (3 * l6 - 12 * l2 * (q04 - 3 * qi04) + 16 * (-q06 + qi06)) * Erf((q0 - qi0) / l)
		  + sqpi * (-3 * l6 + 12 * l2 * (q14 - 3 * qi04) + 16 * (q16 - qi06)) * Erf((q1 - qi0) / l)
		  + 4 * sqpi * (3 * l2 * (q04 - 3 * qi14) + 4 * (q06 - qi16)) * Erf((q0 - qi1) / l)
		  + sqpi * (3 * l6 + 36 * l2 * qi14 + 16 * qi16) * Erf((q1 - qi1) / l)
		  + 3 * l6 * sqpi * Erf((-q0 + qi1) / l) + 12 * l2 * sqpi * q14 * Erf((-q1 + qi1) / l)
		  + 16 * sqpi * q16 * Erf((-q1 + qi1) / l)))
	      / 192.0;

	return I23;
}

double get_I33(double q0, double q1, double qi0, double qi1, double lq)
{
	double I33, q02, q03, q04, q05, q06, q07, q12, q13, q14, q15, q16, q17, l, l2, l4, l3, l5, l6, qi02,
	    qi03, qi04, qi05, qi06, qi07, qi12, qi13, qi14, qi15, qi16, qi17, sqpi, qdiff00, qdiff01, qdiff10,
	    qdiff11;

	sqpi = sqrt(PI);

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;
	l5 = l4 * l;
	l6 = l5 * l;

	q02 = q0 * q0;
	q03 = q02 * q0;
	q04 = q03 * q0;
	q05 = q04 * q0;
	q06 = q05 * q0;
	q07 = q06 * q0;

	q12 = q1 * q1;
	q13 = q12 * q1;
	q14 = q13 * q1;
	q15 = q14 * q1;
	q16 = q15 * q1;
	q17 = q16 * q1;

	qi02 = qi0 * qi0;
	qi03 = qi02 * qi0;
	qi04 = qi03 * qi0;
	qi05 = qi04 * qi0;
	qi06 = qi05 * qi0;
	qi07 = qi06 * qi0;

	qi12 = qi1 * qi1;
	qi13 = qi12 * qi1;
	qi14 = qi13 * qi1;
	qi15 = qi14 * qi1;
	qi16 = qi15 * qi1;
	qi17 = qi16 * qi1;

	qdiff00 = (q0 - qi0) * (q0 - qi0);
	qdiff01 = (q0 - qi1) * (q0 - qi1);
	qdiff10 = (q1 - qi0) * (q1 - qi0);
	qdiff11 = (q1 - qi1) * (q1 - qi1);

	I33 = (l
	       * ((l
		   * (3 * l6 + 3 * l4 * qdiff00
		      - l2 * (16 * q04 + 6 * q03 * qi0 - 9 * q02 * qi02 + 6 * q0 * qi03 + 16 * qi04)
		      - 10 * (q06 + q05 * qi0 + q04 * qi02 + q03 * qi03 + q02 * qi04 + q0 * qi05 + qi06)))
		      * exp(-qdiff00 / l2)
		  + (l
		     * (-3 * l6 - 3 * l4 * qdiff10
			+ l2 * (16 * q14 + 6 * q13 * qi0 - 9 * q12 * qi02 + 6 * q1 * qi03 + 16 * qi04)
			+ 10 * (q16 + q15 * qi0 + q14 * qi02 + q13 * qi03 + q12 * qi04 + q1 * qi05 + qi06)))
			* exp(-qdiff10 / l2)
		  + (l
		     * (-3 * l6 - 3 * l4 * qdiff01
			+ l2 * (16 * q04 + 6 * q03 * qi1 - 9 * q02 * qi12 + 6 * q0 * qi13 + 16 * qi14)
			+ 10 * (q06 + q05 * qi1 + q04 * qi12 + q03 * qi13 + q02 * qi14 + q0 * qi15 + qi16)))
			* exp(-qdiff01 / l2)
		  + (l
		     * (3 * l6 + 3 * l4 * qdiff11
			- l2 * (16 * q14 + 6 * q13 * qi1 - 9 * q12 * qi12 + 6 * q1 * qi13 + 16 * qi14)
			- 10 * (q16 + q15 * qi1 + q14 * qi12 + q13 * qi13 + q12 * qi14 + q1 * qi15 + qi16)))
			* exp(-qdiff11 / l2)
		  - sqpi * (21 * l2 * (q05 - qi05) + 10 * (q07 - qi07)) * Erf((q0 - qi0) / l)
		  + sqpi * (21 * l2 * (q15 - qi05) + 10 * (q17 - qi07)) * Erf((q1 - qi0) / l)
		  + sqpi * (21 * l2 * (q05 - qi15) + 10 * (q07 - qi17)) * Erf((q0 - qi1) / l)
		  + sqpi * qi15 * (21 * l2 + 10 * qi12) * Erf((q1 - qi1) / l)
		  + 21 * l2 * sqpi * q15 * Erf((-q1 + qi1) / l) + 10 * sqpi * q17 * Erf((-q1 + qi1) / l)))
	      / 140.;

	return I33;
}

/* TESTS */
double test_Imn(double qmin, double qmax, double qimin, double qimax, unsigned long nq)
{
	double I22, I22_num, I33, I33_num, I23, I23_num, I32, I32_num, *q, *wq, *qi, *wqi, rel_err, qi2, qi3,
	    q2, q3, Kqqi, lq;
	unsigned long i, j;

	q = malloc(nq * sizeof(double));
	assert(q);
	wq = malloc(nq * sizeof(double));
	assert(wq);
	qi = malloc(nq * sizeof(double));
	assert(qi);
	wqi = malloc(nq * sizeof(double));
	assert(wqi);

	gauss_grid_create(nq, q, wq, qmin, qmax);
	gauss_grid_create(nq, qi, wqi, qimin, qimax);

	lq = 1.0;

	I22_num = 0;
	I23_num = 0;
	I32_num = 0;
	I33_num = 0;
	for (i = 0; i < nq; i++) {
		for (j = 0; j < nq; j++) {

			q2 = q[j] * q[j];
			q3 = q2 * q[j];
			qi2 = qi[i] * qi[i];
			qi3 = qi2 * qi[i];

			Kqqi = exp(-(qi[i] - q[j]) * (qi[i] - q[j]) / (lq * lq));

			I22_num += qi2 * q2 * Kqqi * wqi[i] * wq[j];
			I23_num += qi2 * q3 * Kqqi * wqi[i] * wq[j];
			I32_num += qi3 * q2 * Kqqi * wqi[i] * wq[j];
			I33_num += qi3 * q3 * Kqqi * wqi[i] * wq[j];
		}
	}

	I22 = get_I22(qmin, qmax, qimin, qimax, lq);
	I23 = get_I23(qmin, qmax, qimin, qimax, lq);
	I32 = get_I23(qimin, qimax, qmin, qmax, lq);
	I33 = get_I33(qmin, qmax, qimin, qimax, lq);

	rel_err = 0;
	rel_err += fabs(I22_num - I22) / fabs(I22_num + I22);
	rel_err += fabs(I23_num - I23) / fabs(I23_num + I23);
	rel_err += fabs(I32_num - I32) / fabs(I32_num + I32);
	rel_err += fabs(I33_num - I33) / fabs(I33_num + I33);

	printf("%+.15E %+.15E %+.15E\n", I22, I22_num, fabs(I22 - I22_num));
	printf("%+.15E %+.15E %+.15E\n", I23, I23_num, fabs(I23 - I23_num));
	printf("%+.15E %+.15E %+.15E\n", I32, I32_num, fabs(I32 - I32_num));
	printf("%+.15E %+.15E %+.15E\n", I33, I33_num, fabs(I33 - I33_num));

	free(q);
	free(wq);
	free(qi);
	free(wqi);

	return rel_err;
}
