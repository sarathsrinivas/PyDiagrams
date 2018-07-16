double zs_contact(const double *ke, unsigned int dim, double kf, double g);
int zs_flow(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
	    unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *),
	    double *param, double fac);
/* TESTS */
double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth, unsigned long nphi);
double test_zs_contact(double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi, int seed);
