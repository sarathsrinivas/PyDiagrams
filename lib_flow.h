double zs_contact(const double *ke, unsigned int dim, double kf, double g);
double zsp_contact(const double *ke, unsigned int dim, double kf, double g);
int zs_flow(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
	    unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *),
	    double *param, double fac);
int zsp_flow(double *zsp, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
	     unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *),
	     double *param, double fac);
void get_zs_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dlp, double q, double q_th, double q_phi);
void get_zs_loop_mom_ct(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dlp, double q, double q_th, double q_phi);
void get_zsp_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dl, double q, double q_th, double q_phi);
void get_zsp_loop_mom_ct(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dl, double q, double q_th, double q_phi);
/* TESTS */
double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth, unsigned long nphi);
double test_zs_contact(int isdl_kf, double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
double test_zsp_contact(int isdlp_kf, double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
double test_get_zs_loop_mom(unsigned long ns, double kf, int seed);
double test_get_zsp_loop_mom(unsigned long ns, double kf, int seed);
double test_antisymmetry(double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
double test_convergence(unsigned long ns, double kf, int seed);
