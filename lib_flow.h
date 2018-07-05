double zs_contact(const double *ke, unsigned int dim, double kf, double g);
double *zs_flow(double *ext_mom, unsigned long ns, unsigned int dim1, double kf, unsigned long nq, 
		unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *), unsigned int dim, double *param, double fac);
/* TESTS */
double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth, unsigned long nphi);
