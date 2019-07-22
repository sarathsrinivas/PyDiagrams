#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <lib_io/lib_io.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <lib_pots/lib_pots.h>
#include <atlas/blas.h>
#include "lib_flow.h"

static double get_abs_max(const double *k, unsigned int nk)
{
	unsigned int i;
	double max;

	max = fabs(k[0]);

	for (i = 1; i < nk; i++) {
		if (fabs(k[i]) > max) {
			max = fabs(k[i]);
		}
	}

	return max;
}

void get_regulator_ke_max(double *reg, const double *ke_ct, unsigned long nke, unsigned int dimke,
			  double kmax, double eps)
{

	double dl, dl2, dlp2, P, P2, dlpx, dlp, dlpy, dlpz, Px, Py, Pz, max;
	unsigned long i;

	for (i = 0; i < nke; i++) {

		max = get_abs_max(&ke_ct[dimke * i], dimke);

		reg[i] = fd_reg(max, kmax, eps);
	}
}

void get_regulator_ke_sum(double *reg, const double *ke_ct, unsigned long nke, unsigned int dimke,
			  double kmax, double eps)
{

	double dl, dl2, dlp2, P, P2, dlpx, dlp, dlpy, dlpz, Px, Py, Pz, sum;
	unsigned long i;

	for (i = 0; i < nke; i++) {

		dl = ke_ct[dimke * i + 0];
		dlpx = ke_ct[dimke * i + 1];
		dlpy = ke_ct[dimke * i + 2];
		dlpz = ke_ct[dimke * i + 3];
		Px = ke_ct[dimke * i + 4];
		Py = ke_ct[dimke * i + 5];
		Pz = ke_ct[dimke * i + 6];

		dlp = sqrt(dlpx * dlpx + dlpy * dlpy + dlpz * dlpz);
		P = sqrt(Px * Px + Py * Py + Pz * Pz);

		sum = dl + dlp + P;

		reg[i] = fd_reg(sum, kmax, eps);
	}
}

void get_reg_mat_loop_zs(double *reg1_mat, double *reg2_mat, double kmax, double eps,
			 const double *ke_ct, unsigned long nke, unsigned int dimke,
			 const double *q_ct, unsigned long nq, unsigned int dimq)
{
	double *kl1, *kl2;
	unsigned long i;

	kl1 = malloc(nq * dimke * sizeof(double));
	assert(kl1);
	kl2 = malloc(nq * dimke * sizeof(double));
	assert(kl2);

	for (i = 0; i < nke; i++) {
		get_zs_loop_mom_7d_ct(kl1, kl2, &ke_ct[dimke * i], dimke, q_ct, nq, dimq);

		if (reg1_mat) {
			get_regulator_ke_max(&reg1_mat[nq * i], kl1, nq, dimke, kmax, eps);
		}
		if (reg2_mat) {
			get_regulator_ke_max(&reg2_mat[nq * i], kl2, nq, dimke, kmax, eps);
		}
	}
}

unsigned long get_work_sz_rhs_param(unsigned long nke, unsigned int dimke, unsigned long nq,
				    unsigned long dimq)
{
	unsigned long sz_alloc;

	sz_alloc = 0;

	sz_alloc += nq * dimq; /* q_ct */
	sz_alloc += nke * nke; /* kxx_gma */
	sz_alloc += nq * nq;   /* kxx_fq */
	sz_alloc += nke * nq;  /* A1 */
	sz_alloc += nke * nq;  /* A2 */
	sz_alloc += nke * nke; /* B1 */
	sz_alloc += nke * nke; /* B1 */
	sz_alloc += nke * nq;  /* C */

	sz_alloc += nke * nq; /* Iqe  */
	sz_alloc += nke;      /* IIe */

	sz_alloc += 4 * nq * nq;    /* ktt12 */
	sz_alloc += 2 * nq * nke;   /* ktx12 */
	sz_alloc += 2 * nq * dimke; /* kl12_ct */

	sz_alloc += nq * nke;    /* fqe */
	sz_alloc += nq;		 /* var_fq */
	sz_alloc += 4 * nq * nq; /* var_gma12*/

	sz_alloc += nq * nke;    /* reg12 */
	sz_alloc += 4 * nq * nq; /* reg1x2 */

	sz_alloc += nke; /* var_gma_in */
	sz_alloc += nke; /* var_gma_out */

	return sz_alloc;
}

void init_rhs_param(struct rhs_param *par, double *ke_ct, unsigned long nke, unsigned int dimke,
		    double *q_sph, unsigned long nq, unsigned int dimq, double *pke_ct,
		    double *pq_sph, unsigned long nqr, unsigned long nth, unsigned long nphi,
		    double fac, double kf, unsigned int ke_flag, double reg_mn_max,
		    double reg_mn_eps, double reg_var_max, double reg_var_eps, double *work,
		    unsigned long work_sz)
{
	double *q_ct, *kxx_gma, *kxx_fq, *A1, *A2, *B1, *B2, *C, *Iqe, *IIe, *kl12_ct, *kl12_ct_p,
	    *ktx12, *ktt12, *fqe, *var_fq, *var_gma12, *reg12, *reg1x2, *reg1, *reg2, *reg_kl12,
	    *var_gma_in, *var_gma_out, ALPHA, BETA;
	unsigned long sz_alloc, work_sz_chk, npke, npq;
	int N, LDA, K, INCX, INCY;
	unsigned char UPLO;

	work_sz_chk = get_work_sz_rhs_param(nke, dimke, nq, dimq);

	assert(work);
	assert(work_sz == work_sz_chk && "WORK SIZE DONT MATCH!");

	sz_alloc = 0;

	q_ct = &work[0];
	sz_alloc += nq * dimq;

	kxx_gma = &work[sz_alloc];
	sz_alloc += nke * nke;
	kxx_fq = &work[sz_alloc];
	sz_alloc += nq * nq;

	A1 = &work[sz_alloc];
	sz_alloc += nke * nq;
	A2 = &work[sz_alloc];
	sz_alloc += nke * nq;
	B1 = &work[sz_alloc];
	sz_alloc += nke * nke;
	B2 = &work[sz_alloc];
	sz_alloc += nke * nke;
	C = &work[sz_alloc];
	sz_alloc += nke * nq;

	Iqe = &work[sz_alloc];
	sz_alloc += nke * nq;
	IIe = &work[sz_alloc];
	sz_alloc += nke;

	ktt12 = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;
	ktx12 = &work[sz_alloc];
	sz_alloc += 2 * nq * nke;
	kl12_ct = &work[sz_alloc];
	sz_alloc += 2 * nq * dimke;

	fqe = &work[sz_alloc];
	sz_alloc += nq * nke;
	var_fq = &work[sz_alloc];
	sz_alloc += nq;
	var_gma12 = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;

	reg12 = &work[sz_alloc];
	sz_alloc += nke * nq;
	reg1x2 = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;

	var_gma_in = &work[sz_alloc];
	sz_alloc += nke;
	var_gma_out = &work[sz_alloc];
	sz_alloc += nke;

	assert(sz_alloc == work_sz_chk);

	reg1 = malloc(nq * nke * sizeof(double));
	assert(reg1);
	reg2 = malloc(nq * nke * sizeof(double));
	assert(reg2);
	reg_kl12 = malloc(2 * nq * sizeof(double));
	assert(reg_kl12);

	npke = dimke + 1;
	npq = dimq + 1;

	sph_to_ct(q_ct, q_sph, dimq, nq);

	get_krn_se_ard(kxx_gma, ke_ct, ke_ct, nke, nke, dimke, pke_ct, npke);
	get_krn_se_ard(kxx_fq, q_sph, q_sph, nq, nq, dimq, pq_sph, npq);

	get_zs_covar_Aeq(A1, A2, ke_ct, q_ct, nke, dimke, nq, dimq, pke_ct, npke);
	get_zs_covar_Bes(B1, B2, ke_ct, ke_ct, nke, dimke, pke_ct, npke);
	get_zs_covar_Cqs(C, ke_ct, q_ct, nke, dimke, nq, dimq, pke_ct, npke);

	get_zs_Ifq(Iqe, q_sph, nq, pq_sph, dimq, ke_ct, nke, dimke, nth, fac, kf);

	get_zs_II(IIe, ke_ct, nke, dimke, pq_sph, nth, fac, kf);

	get_zs_loop_mom_7d_ct(kl12_ct, &kl12_ct[nq * dimke], &ke_ct[dimke * ke_flag], dimke, q_ct,
			      nq, dimq);
	get_krn_se_ard(ktx12, kl12_ct, ke_ct, 2 * nq, nke, dimke, pke_ct, npke);
	get_krn_se_ard(ktt12, kl12_ct, kl12_ct, 2 * nq, 2 * nq, dimke, pke_ct, npke);

	get_reg_mat_loop_zs(reg1, reg2, reg_mn_max, reg_mn_eps, ke_ct, nke, dimke, q_ct, nq, dimq);

	N = nq * nke;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	dsbmv_(&UPLO, &N, &K, &ALPHA, reg1, &LDA, reg2, &INCX, &BETA, reg12, &INCY);

	get_regulator_ke_max(reg_kl12, kl12_ct, 2 * nq, dimke, reg_var_max, reg_var_eps);

	N = 2 * nq;
	LDA = 2 * nq;
	INCX = 1;
	INCY = 1;
	ALPHA = 1.0;

	dger_(&N, &N, &ALPHA, reg_kl12, &INCX, reg_kl12, &INCY, reg1x2, &LDA);

	par->ke_ct = ke_ct;
	par->q_sph = q_sph;
	par->q_ct = q_ct;

	par->pke_ct = pke_ct;
	par->pq_sph = pq_sph;

	par->kxx_gma = kxx_gma;
	par->kxx_fq = kxx_fq;

	par->A1 = A1;
	par->A2 = A2;
	par->B1 = B1;
	par->B2 = B2;
	par->C = C;

	par->Iqe = Iqe;
	par->IIe = IIe;

	par->ktt12 = ktt12;
	par->ktx12 = ktx12;
	par->kl12_ct = kl12_ct;

	par->fqe = fqe;
	par->var_fq = var_fq;
	par->var_gma12 = var_gma12;

	par->reg12 = reg12;
	par->reg1x2 = reg1x2;

	par->var_gma_in = var_gma_in;
	par->var_gma_out = var_gma_out;

	par->kf = kf;
	par->ke_flag = ke_flag;
	par->fac = fac;
	par->nq = nq;
	par->nth = nth;
	par->dimke = dimke;
	par->dimq = dimq;

	free(reg1);
	free(reg2);
	free(reg_kl12);
}

unsigned long get_work_sz_rhs_diff_param(unsigned long nke, unsigned int dimke, unsigned long nq,
					 unsigned int dimq)

{
	unsigned long sz_alloc;

	sz_alloc = 0;

	sz_alloc += nq * dimq; /* q_ct */
	sz_alloc += nke * nke; /* kxx_gma_zs */
	sz_alloc += nke * nke; /* kxx_gma_zsp */
	sz_alloc += nq * nq;   /* kxx_fq */

	sz_alloc += nke * nq;  /* A1 */
	sz_alloc += nke * nq;  /* A2 */
	sz_alloc += nke * nke; /* B1 */
	sz_alloc += nke * nke; /* B2 */
	sz_alloc += nke * nq;  /* C */

	sz_alloc += nke * nq;  /* A1p */
	sz_alloc += nke * nq;  /* A2p */
	sz_alloc += nke * nke; /* B1p */
	sz_alloc += nke * nke; /* B2p */
	sz_alloc += nke * nq;  /* Cp */

	sz_alloc += nke * nq; /* Iqe  */
	sz_alloc += nke;      /* IIe */

	sz_alloc += 4 * nq * nq;  /* ktt12_zs */
	sz_alloc += 2 * nq * nke; /* ktx12_zs */

	sz_alloc += 4 * nq * nq;  /* ktt12_zsp */
	sz_alloc += 2 * nq * nke; /* ktx12_zsp */

	sz_alloc += 2 * nq * dimke; /* kl12_ct */

	sz_alloc += nq * nke; /* fqe */
	sz_alloc += nq;       /* var_fq */

	sz_alloc += 4 * nq * nq; /* var_gma12_zs */
	sz_alloc += 4 * nq * nq; /* var_gma12_zsp */
	sz_alloc += 4 * nq * nq; /* var_gma12 */

	return sz_alloc;
}

void init_rhs_diff_param(struct rhs_diff_param *par, double *ke_ct, unsigned long nke,
			 unsigned int dimke, double *q_sph, unsigned long nq, unsigned int dimq,
			 double *pke_ct_zs, double *pke_ct_zsp, double *pq_sph, unsigned long nqr,
			 unsigned long nth, unsigned long nphi, double fac, double kf,
			 unsigned int ke_flag, double *work, unsigned long work_sz)
{
	double *q_ct, *kxx_gma_zs, *kxx_gma_zsp, *kxx_fq, *A1, *A2, *B1, *B2, *C, *A1p, *A2p, *B1p,
	    *B2p, *Cp, *Iqe, *IIe, *kl12_ct, *kl12_ct_p, *ktx12_zs, *ktt12_zs, *ktx12_zsp,
	    *ktt12_zsp, *fqe, *var_fq, *var_gma12_zs, *var_gma12_zsp, *var_gma12;
	unsigned long sz_alloc, work_sz_chk, npke, npq;

	work_sz_chk = get_work_sz_rhs_diff_param(nke, dimke, nq, dimq);

	assert(work);
	assert(work_sz == work_sz_chk && "WORK SIZE DONT MATCH!");

	sz_alloc = 0;

	q_ct = &work[0];
	sz_alloc += nq * dimq;

	kxx_gma_zs = &work[sz_alloc];
	sz_alloc += nke * nke;
	kxx_gma_zsp = &work[sz_alloc];
	sz_alloc += nke * nke;

	kxx_fq = &work[sz_alloc];
	sz_alloc += nq * nq;

	A1 = &work[sz_alloc];
	sz_alloc += nke * nq;
	A2 = &work[sz_alloc];
	sz_alloc += nke * nq;
	B1 = &work[sz_alloc];
	sz_alloc += nke * nke;
	B2 = &work[sz_alloc];
	sz_alloc += nke * nke;
	C = &work[sz_alloc];
	sz_alloc += nke * nq;

	A1p = &work[sz_alloc];
	sz_alloc += nke * nq;
	A2p = &work[sz_alloc];
	sz_alloc += nke * nq;
	B1p = &work[sz_alloc];
	sz_alloc += nke * nke;
	B2p = &work[sz_alloc];
	sz_alloc += nke * nke;
	Cp = &work[sz_alloc];
	sz_alloc += nke * nq;

	Iqe = &work[sz_alloc];
	sz_alloc += nke * nq;
	IIe = &work[sz_alloc];
	sz_alloc += nke;

	ktt12_zs = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;
	ktx12_zs = &work[sz_alloc];
	sz_alloc += 2 * nq * nke;

	ktt12_zsp = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;
	ktx12_zsp = &work[sz_alloc];
	sz_alloc += 2 * nq * nke;

	kl12_ct = &work[sz_alloc];
	sz_alloc += 2 * nq * dimke;

	fqe = &work[sz_alloc];
	sz_alloc += nq * nke;
	var_fq = &work[sz_alloc];
	sz_alloc += nq;

	var_gma12_zs = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;
	var_gma12_zsp = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;
	var_gma12 = &work[sz_alloc];
	sz_alloc += 4 * nq * nq;

	npke = dimke + 1;
	npq = dimq + 1;

	sph_to_ct(q_ct, q_sph, dimq, nq);

	get_krn_se_ard(kxx_gma_zs, ke_ct, ke_ct, nke, nke, dimke, pke_ct_zs, npke);
	get_krn_se_ard(kxx_gma_zsp, ke_ct, ke_ct, nke, nke, dimke, pke_ct_zsp, npke);

	get_krn_se_ard(kxx_fq, q_sph, q_sph, nq, nq, dimq, pq_sph, npq);

	get_zs_covar_Aeq(A1, A2, ke_ct, q_ct, nke, dimke, nq, dimq, pke_ct_zs, npke);
	get_zs_covar_Bes(B1, B2, ke_ct, ke_ct, nke, dimke, pke_ct_zs, npke);
	get_zs_covar_Cqs(C, ke_ct, q_ct, nke, dimke, nq, dimq, pke_ct_zs, npke);

	get_zs_covar_Aeq(A1, A2, ke_ct, q_ct, nke, dimke, nq, dimq, pke_ct_zsp, npke);
	get_zs_covar_Bes(B1, B2, ke_ct, ke_ct, nke, dimke, pke_ct_zsp, npke);
	get_zs_covar_Cqs(C, ke_ct, q_ct, nke, dimke, nq, dimq, pke_ct_zsp, npke);

	get_zs_Ifq(Iqe, q_sph, nq, pq_sph, dimq, ke_ct, nke, dimke, nth, fac, kf);

	get_zs_II(IIe, ke_ct, nke, dimke, pq_sph, nth, fac, kf);

	get_zs_loop_mom_7d_ct(kl12_ct, &kl12_ct[nq * dimke], &ke_ct[dimke * ke_flag], dimke, q_ct,
			      nq, dimq);

	get_krn_se_ard(ktx12_zs, kl12_ct, ke_ct, 2 * nq, nke, dimke, pke_ct_zs, npke);
	get_krn_se_ard(ktt12_zs, kl12_ct, kl12_ct, 2 * nq, 2 * nq, dimke, pke_ct_zs, npke);

	get_krn_se_ard(ktx12_zsp, kl12_ct, ke_ct, 2 * nq, nke, dimke, pke_ct_zsp, npke);
	get_krn_se_ard(ktt12_zsp, kl12_ct, kl12_ct, 2 * nq, 2 * nq, dimke, pke_ct_zsp, npke);

	par->ke_ct = ke_ct;
	par->q_sph = q_sph;
	par->q_ct = q_ct;

	par->pke_ct_zs = pke_ct_zs;
	par->pke_ct_zsp = pke_ct_zsp;
	par->pq_sph = pq_sph;

	par->kxx_gma_zs = kxx_gma_zs;
	par->kxx_gma_zsp = kxx_gma_zsp;
	par->kxx_fq = kxx_fq;

	par->A1 = A1;
	par->A2 = A2;
	par->B1 = B1;
	par->B2 = B2;
	par->C = C;

	par->A1p = A1p;
	par->A2p = A2p;
	par->B1p = B1p;
	par->B2p = B2p;
	par->Cp = Cp;

	par->Iqe = Iqe;
	par->IIe = IIe;

	par->ktt12_zs = ktt12_zs;
	par->ktx12_zs = ktx12_zs;

	par->ktt12_zsp = ktt12_zsp;
	par->ktx12_zsp = ktx12_zsp;

	par->kl12_ct = kl12_ct;

	par->fqe = fqe;
	par->var_fq = var_fq;

	par->var_gma12_zs = var_gma12_zs;
	par->var_gma12_zsp = var_gma12_zsp;
	par->var_gma12 = var_gma12;

	par->kf = kf;
	par->ke_flag = ke_flag;
	par->fac = fac;
	par->nq = nq;
	par->nth = nth;
	par->dimke = dimke;
	par->dimq = dimq;
}

void test_get_abs_max(unsigned int n, int seed)
{

	unsigned int i;
	double max, *k;
	dsfmt_t drng;

	k = malloc(n * sizeof(double));
	assert(k);

	dsfmt_init_gen_rand(&drng, seed);

	for (i = 0; i < n; i++) {
		k[i] = 0.5 - dsfmt_genrand_close_open(&drng);
		fprintf(stderr, "%+.15E\n", k[i]);
	}

	max = get_abs_max(k, n);

	fprintf(stderr, "MAX: %+.15E\n\n", max);

	free(k);
}
