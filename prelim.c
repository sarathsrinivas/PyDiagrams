#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <lib_io/lib_io.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <lib_pots/lib_pots.h>
#include "lib_flow.h"

void get_regulator_ke_max(double *reg, const double *ke_ct, unsigned long nke, unsigned int dimke,
			  double kmax, double eps)
{

	double dl, dl2, dlp2, P, P2, dlpx, dlp, dlpy, dlpz, Px, Py, Pz, max;
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

		max = (dlp > P) ? dlp : P;

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

	return sz_alloc;
}

void init_rhs_param(struct rhs_param *par, double *ke_ct, unsigned long nke, unsigned int dimke,
		    double *q_sph, unsigned long nq, unsigned int dimq, double *pke_ct,
		    double *pq_sph, unsigned long nqr, unsigned long nth, unsigned long nphi,
		    double fac, double kf, unsigned int ke_flag, double *work,
		    unsigned long work_sz)
{
	double *q_ct, *kxx_gma, *kxx_fq, *A1, *A2, *B1, *B2, *C, *Iqe, *IIe, *kl12_ct, *kl12_ct_p,
	    *ktx12, *ktt12, *fqe, *var_fq, *var_gma12;
	unsigned long sz_alloc, work_sz_chk, npke, npq;

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

	par->kf = kf;
	par->ke_flag = ke_flag;
	par->fac = fac;
	par->nq = nq;
	par->nth = nth;
	par->dimke = dimke;
	par->dimq = dimq;
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
