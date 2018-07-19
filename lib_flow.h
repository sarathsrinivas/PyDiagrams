double zs_contact(const double *ke, unsigned int dim, double kf, double g);
int zs_flow(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
	    unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *),
	    double *param, double fac);
void get_zs_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dlp, double q, double q_th, double q_phi);
/* TESTS */
double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth, unsigned long nphi);
double test_zs_contact(double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
double test_get_zs_loop_mom(unsigned long ns, double kf, int seed);
