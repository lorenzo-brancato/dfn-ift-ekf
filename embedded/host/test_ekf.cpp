// EKF against the Python golden run (CYC20 from a start error): SOC within --tol % (default 0.1), covariance
// positive definite (Cholesky of a copy every --pd-every steps), time per step.
#include <cmath>
#include <cstring>
#include "common.h"

static dfn::EKF ekf;

int main(int argc, char** argv) {
    double tol = 0.1;
    const char* err = "+0.40";
    int pd_every = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--err") && i + 1 < argc) err = argv[++i];
        else if (!std::strcmp(argv[i], "--pd-every") && i + 1 < argc) pd_every = std::atoi(argv[++i]);
    }
    Csv c;
    std::string file = golden(std::string("ekf_CYC20_e") + err + ".csv");
    if (!read_csv(file, c)) { std::printf("no golden %s\n", file.c_str()); return 0; }
    std::vector<double> I = c.get("I"), y = c.get("y"), Vp = c.get("V"), socp = c.get("soc"), sdp = c.get("sd_soc"),
                        Rfp = c.get("Rf"), soc_ref = c.get("soc_ref");
    int idt = -1;
    for (int d = 0; d < dfn_model::NDT; ++d) {
        double dt = (c.rows.size() > 1) ? c.rows[1][0] - c.rows[0][0] : 0;
        if (std::fabs(dfn_model::DT[d] - dt) < 1e-6) idt = d;
    }
    if (idt < 0) { std::printf("step length of the golden run not in the model's DT\n"); return 2; }
    real_t z0[dfn::NZ];
    z_at(dfn_model::q0 + std::atof(err), z0);
    ekf.init(z0, 0, idt, (real_t)I[0]);
    const size_t n = I.size();
    double smax = 0, s2 = 0, vmax = 0, rfmax = 0, sdmax = 0, pdmin = 1e300, mdmin = 1e300, e2ref = 0;
    int npd_bad = 0;
    Timer tm;
    double t_pd = 0;
    for (size_t k = 0; k < n; ++k) {
        dfn::EkfOut o = ekf.step((real_t)I[k], (real_t)y[k], idt);
        double ds = std::fabs(o.soc - socp[k]) * 100;
        if (!(ds <= smax)) smax = ds;
        s2 += ds * ds;
        double dv = std::fabs(o.V - Vp[k]) * 1e3;
        if (!(dv <= vmax)) vmax = dv;
        double dr = std::fabs(o.Rf - Rfp[k]);
        if (!(dr <= rfmax)) rfmax = dr;
        double dsd = std::fabs(o.sd_soc - sdp[k]) * 100;
        if (dsd > sdmax) sdmax = dsd;
        if (k >= n / 2) e2ref += (o.soc - soc_ref[k]) * (o.soc - soc_ref[k]) * 1e4;
        if (!(o.min_diag >= mdmin)) mdmin = o.min_diag;                // NaN sticks
#if DFN_COV_UD
        // UD: P = U D U^T is positive definite iff every D_j > 0 (checked every step; min_diag is min D)
        if (!(o.min_diag > 0)) ++npd_bad;
        (void)t_pd; (void)pd_every;
#else
        if (pd_every > 0 && (k % pd_every == 0 || k == n - 1)) {      // Joseph: Cholesky of a copy
            Timer tp;
            static double Pd[dfn::NX][dfn::NX];
            for (int i = 0; i < dfn::NX; ++i)
                for (int j = 0; j < dfn::NX; ++j) Pd[i][j] = ekf.P[i][j];
            double p = chol_min(Pd);
            t_pd += tp.s();
            if (!(p > 0)) ++npd_bad;
            if (p < pdmin) pdmin = p;
        }
#endif
    }
    double wall = tm.s() - t_pd;
    bool ok = smax <= tol && npd_bad == 0;
    std::printf("EKF  %s  %s  CYC20 e%s  %zu steps of %g s\n", dfn_model::TAG, prec(), err, n, dfn_model::DT[idt]);
    std::printf("  SOC - SOC_py: max %.4f %% rms %.4f %% (tol %.2f)   V - V_py max %.3f mV   Rf - Rf_py max %.3g   sd_SOC diff max %.4f %%\n",
                smax, std::sqrt(s2 / n), tol, vmax, rfmax, sdmax);
    std::printf("  SOC vs Coulomb counting, 2nd half rms %.3f %%   final SOC %.4f (py %.4f)  Rf %.3f (py %.3f)\n",
                std::sqrt(e2ref / (n - n / 2)), (double)dfn::soc_of_ybar(dfn::ybar(ekf.x)), socp[n - 1],
                (double)std::exp(ekf.x[dfn::ITH]), Rfp[n - 1]);
#if DFN_COV_UD
    std::printf("  covariance (UD): min D %.3g, D <= 0 at %d steps\n", mdmin, npd_bad);
#else
    std::printf("  covariance (Joseph): min diag %.3g, min Cholesky pivot^2 %.3g, not PD at %d checks (every %d steps)\n",
                mdmin, pdmin, npd_bad, pd_every);
#endif
    std::printf("  %.3f ms/step   %s\n", 1e3 * wall / n, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
