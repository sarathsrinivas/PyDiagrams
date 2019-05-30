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
