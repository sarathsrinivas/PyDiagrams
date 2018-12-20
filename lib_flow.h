double zs_contact(const double *ke, unsigned int dim, double kf, double g);
double zsp_contact(const double *ke, unsigned int dim, double kf, double g);
int zs_flow(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
	    unsigned long nth, unsigned long nphi, double (*vfun)(double *, unsigned int, double *),
	    double *param, double fac);
int zsp_flow(double *zsp, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
	     unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *),
	     double *param, double fac);
void get_zs_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dlp, double q, double q_th, double q_phi);
void get_zs_loop_mom_ct(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dlp, double q, double q_th, double q_phi);
void get_zsp_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dl, double q, double q_th, double q_phi);
void get_zsp_loop_mom_ct(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dl, double q, double q_th, double q_phi);
void fill_zs_loop_mom_ct(double *kl1, double *kl2, unsigned int dim, double *ke, double *qvec, unsigned long nq);
double get_zs_energy(const double *ke, unsigned int dim);
double get_zsp_energy(const double *ke, unsigned int dim);
void sph_ct_mom6(const double *ke, unsigned int dim, unsigned long ns, double *ke_ct);
void get_zs_reg_limits(double kmax, const double *ke, double phi_dlp, double th_q, double phi_q,  double *lims);
/* Phase Spaces */
void get_zs_th_grid(double *th, double *wth, double *qmin, double *qmax, unsigned long nth, const double *gth, const double *gwth, double dl, double kf, double fac);
unsigned long get_ph_phase_space(double kf, double dl, unsigned long nq, unsigned long nth,
				 unsigned long nphi, double fac_th_brk, double *gl_q, double *wl_q,
				 double *gl_th, double *wl_th, double *gl_phi, double *wl_phi, double *qvec,
				 double *wt, double *work_mem, unsigned long nwork);
double get_ph_space_vol_fd(double dl, double kf, unsigned long nq, unsigned long nth, unsigned long nphi);
double get_ph_space_vol(double dl, double kf, unsigned long nq, unsigned long nth, unsigned long nphi);
double get_ph_vol_exct(double dl, double kf);
/* GPR FLOW RHS */
int zs_flow_gpr(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, double kmax, unsigned long nq,
		unsigned long nth, unsigned long nphi, double fac, const double *p, int np, const double *wt, double *lkrxx);
int zsp_flow_gpr(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
		unsigned long nth, unsigned long nphi, double fac, const double *p, int np, const double *wt);
/* TESTS */
double test_ph_phase_space_vol(double dl, double kf, double fac_th_brk, unsigned long nq, unsigned long nth, unsigned long nphi);
double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth, unsigned long nphi);
double test_zs_contact(int isdl_kf, double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
double test_zsp_contact(int isdlp_kf, double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
double test_get_zs_loop_mom(unsigned long ns, double kf, int seed);
double test_get_zsp_loop_mom(unsigned long ns, double kf, int seed);
double test_antisymmetry(double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
double test_convergence(unsigned long ns, double kf, int seed);
double test_mom_closure(double kmax, unsigned long ns, int seed);
/* GPR TESTS */
double test_gpr_fit(unsigned long ns, unsigned long nt, double kf, int seed);
/* FLOW F(q) */
double Erf(double x);
void get_I2q(double *I2q, const double *xq, unsigned long nq, unsigned int dim, double *q0, double *q1, unsigned long nth, double lq);
void get_I3q(double *I3q, const double *xq, unsigned long nq, unsigned int dim, double *q0, double *q1, unsigned long nth, double lq);
void get_Ifq(double *Ifq, const double *xq, unsigned long nq, const double *l, unsigned int dim, const double *e_ext, const double *th_max, unsigned long nke);
double test_get_I2q(unsigned int tn, double q0, double q1, double lq);
double test_get_I3q(unsigned int tn, double q0, double q1, double lq);
