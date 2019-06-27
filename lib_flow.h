/* DATA STRUCTURES */
struct rhs_param {
	double *ke_ct, *q_ct, *q_sph, *pke_ct, *pq_sph, *kxx_gma, *kxx_fq, *ktt12, *ktx12, *kl12_ct,
	    *A1, *B1, *A2, *B2, *C, *Iqe, *IIe, *fqe, *var_fq, *var_gma12, *reg12, *reg1x2;
	double fac, kf;
	unsigned int dimke, dimq, ke_flag;
	unsigned long nq, nth;
};

struct rhs_diff_param {
	double *ke_ct, *q_ct, *q_sph, *pke_ct_zs, *pke_ct_zsp, *pq_sph, *kxx_gma_zs, *kxx_gma_zsp,
	    *kxx_fq, *ktt12_zs, *ktx12_zs, *kl12_ct, *ktt12_zsp, *ktx12_zsp, *var_gma12_zsp,
	    *var_gma12_zs, *var_gma12, *A1, *B1, *A2, *B2, *C, *A1p, *B1p, *A2p, *B2p, *Cp, *Iqe,
	    *IIe, *fqe, *var_fq;
	double fac, kf;
	unsigned int dimke, dimq, ke_flag;
	unsigned long nq, nth;
};

struct rhs_ph_param {
	double *ke_ct, *kep_ct, *q_ct, *q_sph, *pke_ct, *pq_sph, *kxx_gma, *kxx_fq, *ktt12, *ktx12,
	    *kl12_ct, *A1, *B1, *A2, *B2, *C, *Iqe, *IIe, *kxxp_gma, *ktt12p, *ktx12p, *kl12p_ct,
	    *A1p, *B1p, *A2p, *B2p, *Cp, *Iqep, *IIep, *fqe, *var_fq, *fqep, *var_fqp, *var_gma12,
	    *var_gma12p;
	double fac, kf;
	unsigned int dimke, dimq, ke_flag;
	unsigned long nq, nth;
};

double zs_contact(const double *ke, unsigned int dim, double kf, double g);
double zsp_contact(const double *ke, unsigned int dim, double kf, double g);
void zs_flow(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf,
	     unsigned long nq, unsigned long nth, unsigned long nphi,
	     double (*vfun)(double *, unsigned int, double *), double *param, double fac);
int zsp_flow(double *zsp, double *ext_mom, unsigned long ns, unsigned int dim, double kf,
	     unsigned long nq, unsigned long nth, unsigned long nphi,
	     double vfun(double *, unsigned int, double *), double *param, double fac);
void get_zs_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dlp,
		     double q, double q_th, double q_phi);
void get_zs_loop_mom_ct(double *kl1, double *kl2, unsigned int dim, const double *ke,
			double phi_dlp, double q, double q_th, double q_phi);
void get_zsp_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dl,
		      double q, double q_th, double q_phi);
void get_zsp_loop_mom_ct(double *kl1, double *kl2, unsigned int dim, const double *ke,
			 double phi_dl, double q, double q_th, double q_phi);
void fill_zs_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke,
		      const double *qvec, unsigned long nq);
double get_zs_energy(const double *ke, unsigned int dim);
double get_zsp_energy(const double *ke, unsigned int dim);
void sph_ct_mom6_zs(double *ke_ct, const double *ke, unsigned int dim, unsigned long ns);
void get_zs_reg_limits(double kmax, const double *ke, double phi_dlp, double th_q, double phi_q,
		       double *lims);

/* Phase Spaces */
void get_zs_th_grid(double *th, double *wth, double *qmin, double *qmax, unsigned long nth,
		    const double *gth, const double *gwth, double dl, double kf, double fac);
void get_ph_space_grid(double *xq, double *wxq, unsigned int dimq, double dl, double kf,
		       unsigned long nq, unsigned long nth, unsigned long nphi);
double get_ph_space_vol(double dl, double kf, unsigned long nq, unsigned long nth,
			unsigned long nphi);
double get_ph_space_vol_fd(double dl, double kf, unsigned long nq, unsigned long nth,
			   unsigned long nphi);
double get_ph_vol_exct(double dl, double kf);

/* GPR FLOW RHS */
int zs_flow_gpr(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf,
		double kmax, unsigned long nq, unsigned long nth, unsigned long nphi, double fac,
		const double *p, int np, const double *wt, double *lkrxx);
int zsp_flow_gpr(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf,
		 unsigned long nq, unsigned long nth, unsigned long nphi, double fac,
		 const double *p, int np, const double *wt);

/* TESTS */
/*
double test_ph_phase_space_vol(double dl, double kf, double fac_th_brk, unsigned long nq, unsigned
long nth, unsigned long nphi);
*/
double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth,
			   unsigned long nphi);
double test_zs_contact(int isdl_kf, double kf, unsigned long ns, unsigned long nq,
		       unsigned long nth, unsigned long nphi, int seed);
double test_zsp_contact(int isdlp_kf, double kf, unsigned long ns, unsigned long nq,
			unsigned long nth, unsigned long nphi, int seed);
double test_get_zs_loop_mom(unsigned long ns, double kf, int seed);
double test_get_zsp_loop_mom(unsigned long ns, double kf, int seed);
double test_antisymmetry(double kf, unsigned long ns, unsigned long nq, unsigned long nth,
			 unsigned long nphi, int seed);
double test_convergence(unsigned long ns, double kf, int seed);
double test_mom_closure(double kmax, unsigned long ns, int seed);

/* GPR TESTS */
double test_gpr_fit(unsigned long ns, unsigned long nt, double kf, int seed);

/* MOMENTUM SAMPLING */
void fill_ke_sample_zs_ct_box(double *ke_ct, unsigned long ns, double *st, double *en,
			      unsigned int seed);
void fill_ke_sample_zs_ct(double *ke_ct, unsigned long ns, double *st, double *en,
			  unsigned int seed);
void fill_ke_sample_zs_ct_exp(double *ke_ct, unsigned long ns, double *st, double *en, double sig,
			      double a, unsigned int seed);
void get_kep_sample_zsp_ct(double *kep_ct, const double *ke_ct, unsigned long nke,
			   unsigned int dim);
void fill_q_sample(double *ke, unsigned long ns, double st, double en, unsigned int seed);
void sph_to_ct(double *q_ct, const double *q, unsigned int dimq, unsigned long nq);
void print_mom(const double *k, unsigned long nk, unsigned int dimk, FILE *out);

/* TESTS */
double test_zs_zsp_rot(unsigned long nke, int seed);

/* FLOW FUNS */
double get_zs_energy_7d_ct(const double *ke_ct, unsigned int dim);
void get_zs_loop_mom_7d_ct(double *kl1_ct, double *kl2_ct, const double *ke_ct, unsigned int dimke,
			   const double *q_ct, unsigned long nq, unsigned int dimq);
void get_zs_num_7d_ct(double *zs_ct, double *ke_ct, unsigned long nke, unsigned int dimke,
		      double kf, unsigned long nq, unsigned long nth, unsigned long nphi,
		      double (*vfun)(double *, unsigned int, double *), double *param);

/* FLOW NUMERICAL */
double get_zs_contact(double g, double kf, double *ke_ct, unsigned int dimke);
void get_zs_num(double *zs, double *ke_ct, unsigned long nke, unsigned int dimke, double kf,
		unsigned long nq, unsigned long nth, unsigned long nphi,
		double (*vfun)(double *, unsigned int, double *), double *param);
void get_rhs_num(double *rhs, double *ke_ct, unsigned long nke, unsigned int dimke, double kf,
		 unsigned long nq, unsigned long nth, unsigned long nphi,
		 double (*vfun)(double *, unsigned int, double *), double *param);

/* TESTS */
double test_get_zs_num(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi,
		       int seed);
double test_get_zsp_num(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi,
			int seed);
double test_rhs_antisymmetry(unsigned long nke, int seed);

/* FLOW F(Q) */
double Erf(double x);
void get_zs_fq_mat_fun(double *fqke, const double *ke, unsigned long nke, unsigned int dimke,
		       const double *qi, unsigned long nqi, unsigned int dimq,
		       double v_fun(double *, unsigned int, double *), double *vpar);
void get_fq_weight_mat(double *wtqke, double *lkqq, const double *kqq, const double *fqke,
		       unsigned long nke, unsigned long nqi);
void get_I2q(double *I2q, const double *xq, unsigned long nq, unsigned int dim, double *q0,
	     double *q1, unsigned long nth, double lq);
void get_I3q(double *I3q, const double *xq, unsigned long nq, unsigned int dim, double *q0,
	     double *q1, unsigned long nth, double lq);
void get_zs_Ifq(double *Ifq, const double *xq, unsigned long nq, const double *l, unsigned int dimq,
		const double *ke_ct, unsigned long nke, unsigned int dimke, unsigned long nth,
		double fac, double kf);
void get_zs_num(double *zs, double *ke_ct, unsigned long nke, unsigned int dimke, double kf,
		unsigned long nq, unsigned long nth, unsigned long nphi,
		double (*vfun)(double *, unsigned int, double *), double *param);
double get_zs_contact(double g, double kf, double *ke_ct, unsigned int dimke);
void get_zs_Ifq_num(double *Ifq_num, double *ke_ct, unsigned long nke, unsigned int dimke,
		    double kf, unsigned long nq, unsigned long nth, unsigned long nphi, double *xqi,
		    unsigned long nxqi, unsigned int dimq, double *pq, double fac);
void predict_zs_fq(double *zs, unsigned long nke, const double *wq, unsigned long nq,
		   const double *Ifq);

/* TESTS FOR FLOW F(Q) */
double test_get_I2q(unsigned int nqi, double q0, double q1, double lq);
double test_get_I3q(unsigned int nqi, double q0, double q1, double lq);
double test_Ifq(unsigned long nke, unsigned long nqi, unsigned long nth, double fac, double kmax,
		double kf, int seed);
double test_get_zs_num(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi,
		       int seed);

/* FLOW F(Q) VARIANCE FUNCTIONS */
double get_I22(double q0, double q1, double qi0, double qi1, double lq);
double get_I23(double q0, double q1, double qi0, double qi1, double lq);
double get_I33(double q0, double q1, double qi0, double qi1, double lq);
void get_zs_II(double *II, const double *ke_ct, unsigned long nke, unsigned int dimke,
	       const double *lxq, unsigned long nth, double fac, double kf);
double get_integ_covar(const double *Iq, const double *kqq_chlsk, unsigned long nxq,
		       double *tmp_vec);
void get_zs_II_num(double *II, const double *ke_ct, unsigned long nke, unsigned int dimke,
		   const double *lxq, unsigned int dimq, unsigned long nq, unsigned long nth,
		   unsigned long nphi, double fac, double kf);

/* TESTS FOR FLOW F(Q) VARIANCE */
double test_Imn(double qmin, double qmax, double qimin, double qimax, unsigned long nq);
double test_get_integ_covar(unsigned long n, int seed);
double test_get_zs_II(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi,
		      double kmax, double kf, int seed);

/* GMA  PRECOMPUTED COVARIANCE */

void get_zs_covar_Aeq(double *A1, double *A2, const double *ke_ct, const double *q_ct,
		      unsigned long nke, unsigned int dimke, unsigned long nq, unsigned int dimq,
		      const double *pke, unsigned long np);
void get_zs_covar_Bes(double *B1, double *B2, const double *ke_ct, const double *ks_ct,
		      unsigned long nke, unsigned int dimke, const double *pke, unsigned long np);
void get_zs_covar_Cqs(double *C, const double *ke_ct, const double *q_ct, unsigned long nke,
		      unsigned int dimke, unsigned long nq, unsigned int dimq, const double *pke,
		      unsigned long np);

/*  TESTS FOR GMA  PRECOMPUTED COVARIANCE */
double test_zs_gma_covar(unsigned long nke, unsigned long nq, int seed);

/* STAGE ONE */
void interpolate_gma(double *gma, const double *wt_gma, const double *Aeq, const double *Bes,
		     const double *Csq, unsigned long nq, unsigned long nke);
void get_fq_samples(double *fq, double *var_fq, const double *wt_gma, const double *A1eq,
		    const double *B1es, const double *A2eq, const double *B2es, const double *Csq,
		    const double *var_gma12, unsigned int ke_flag, unsigned long nq,
		    unsigned long nke);
void get_fq_samples_reg(double *fq_reg, double *var_fq, const double *wt_gma, const double *A1,
			const double *B1, const double *A2, const double *B2, const double *C,
			double *var_gma12, const double *reg12, const double *reg1x2,
			unsigned int ke_flag, unsigned long nq, unsigned long nke);
void get_fq_as_samples(double *fq, double *var_fq, const double *wt_gma_zs,
		       const double *wt_gma_zsp, const double *A1, const double *A2,
		       const double *B1, const double *B2, const double *C, const double *A1p,
		       const double *A2p, const double *B1p, const double *B2p, const double *Cp,
		       const double *var_gma12, unsigned int ke_flag, unsigned long nq,
		       unsigned long nke);
void get_var_fq(double *var_fq, const double *gma1, const double *gma2, const double *var_gma12,
		unsigned long nq);

/* TESTS FOR STAGE ONE */
double test_get_zs_fq_samples(unsigned long nke, unsigned long nq, int seed);
double test_get_zs_fq_as_samples(unsigned long nke, unsigned long nq, int seed);

/* STAGE TWO */
void get_noise_covar_chd(double *lknxx, const double *kxx, const double *var, unsigned long nx);
void get_fq_weights(double *wt_fq, const double *lknqq, unsigned long nq, unsigned long nke);
void get_fq_weights_2(double *wt_fq, const double *lknqq, const double *fq, unsigned long nq,
		      unsigned long nke);
void get_gma_gpr_mean(double *gma_gpr, const double *Ifq, const double *wfq, unsigned long nke,
		      unsigned long nq);
void get_gma_gpr_var(double *var_gma_gpr, const double *II, const double *Iqe, const double *lknqq,
		     unsigned long nq, unsigned long nke);
void get_gma_weight(double *wt_gma, const double *lknxx_gma, const double *gma, unsigned long nke);
void get_noisy_inverse(double *wt, const double *lkqq, const double *var, const double *y,
		       unsigned long nq, unsigned long nke);

/* TESTS FOR STAGE TWO */
double test_noisy_inverse(unsigned long nq, unsigned long nke, double sigma2, int seed);

/* PRELIMINARY COMPUTATIONS */
void get_regulator_ke_max(double *reg, const double *ke_ct, unsigned long nke, unsigned int dimke,
			  double kmax, double eps);
void get_regulator_ke_sum(double *reg, const double *ke_ct, unsigned long nke, unsigned int dimke,
			  double kmax, double eps);
void get_reg_mat_loop_zs(double *reg1_mat, double *reg2_mat, double kmax, double eps,
			 const double *ke_ct, unsigned long nke, unsigned int dimke,
			 const double *q_ct, unsigned long nq, unsigned int dimq);
unsigned long get_work_sz_rhs_param(unsigned long nke, unsigned int dimke, unsigned long nq,
				    unsigned long dimq);
void init_rhs_param(struct rhs_param *par, double *ke_ct, unsigned long nke, unsigned int dimke,
		    double *q_sph, unsigned long nq, unsigned int dimq, double *pke_ct,
		    double *pq_sph, unsigned long nqr, unsigned long nth, unsigned long nphi,
		    double fac, double kf, unsigned int ke_flag, double reg_mn_max,
		    double reg_mn_eps, double reg_var_max, double reg_var_eps, double *work,
		    unsigned long work_sz);
unsigned long get_work_sz_rhs_diff_param(unsigned long nke, unsigned int dimke, unsigned long nq,
					 unsigned int dimq);
void init_rhs_diff_param(struct rhs_diff_param *par, double *ke_ct, unsigned long nke,
			 unsigned int dimke, double *q_sph, unsigned long nq, unsigned int dimq,
			 double *pke_ct_zs, double *pke_ct_zsp, double *pq_sph, unsigned long nqr,
			 unsigned long nth, unsigned long nphi, double fac, double kf,
			 unsigned int ke_flag, double *work, unsigned long work_sz);

/* TESTS */
void test_get_abs_max(unsigned int n, int seed);

/* RHS */
void get_rhs_block(double *gma, double *var_gma, const double *gma0, const double *var_gma0,
		   unsigned long nke, struct rhs_param *par);
void get_rhs_diff_block(double *gma, double *var_gma, const double *gma0_zs,
			const double *var_gma0_zs, const double *gma0_zsp,
			const double *var_gma0_zsp, unsigned long nke, struct rhs_diff_param *par);
void flow_rhs(double *gma, double *var_gma, double *gma0, double *var_gma0, unsigned long nke,
	      void *param);
void flow_rhs_ph(double *gma, double *var_gma, double *gma0, double *var_gma0, unsigned long nke,
		 void *param);
