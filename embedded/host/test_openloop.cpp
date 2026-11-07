// Open loop against the Python golden files: V within --tol mV (default 0.1 in double, 1 in float), time per step.
#include <cmath>
#include "common.h"

static dfn::OpenLoop ol;

static int run(const char* test, double tol_mV) {
    Csv c;
    if (!read_csv(golden(std::string("openloop_") + test + ".csv"), c)) {
        std::printf("  %-7s no golden file, skipped\n", test);
        return 0;
    }
    const double q = read_scalar(golden(std::string("openloop_") + test + ".start"));
    std::vector<double> I = c.get("I"), Vp = c.get("V"), Vm = c.get("V_meas"), idt = c.get("idt"), socp = c.get("soc");
    real_t z0[dfn::NZ];
    z_at(q, z0);
    ol.reset(z0);
    const size_t n = I.size();
    double emax = 0, e2 = 0, rm2 = 0, smax = 0, resmax = 0;
    long nit = 0;
    Timer tm;
    for (size_t k = 0; k < n; ++k) {
        double V = ol.step((real_t)I[k], (int)idt[k]);
        double e = std::fabs(V - Vp[k]);
        if (!(e <= emax)) emax = e;
        e2 += e * e;
        rm2 += (V - Vm[k]) * (V - Vm[k]);
        double ds = std::fabs(dfn::soc_of_ybar(dfn::ybar(ol.z)) - (k + 1 < n ? socp[k + 1] : socp[k]));
        if (k + 1 < n && ds > smax) smax = ds;
        if ((double)ol.res_last > resmax) resmax = (double)ol.res_last;
        nit += ol.cl.nit_last;
    }
    double wall = tm.s();
    bool ok = emax * 1e3 <= tol_mV;
    std::printf("  %-7s %6zu steps  |V - V_py| max %.4f mV rms %.4f mV  (tol %.2f)  RMSE vs meas %.2f mV  SOC max diff %.1e"
                "  max||F|| %.1e  Newton it/step %.2f  %.3f ms/step  %s\n",
                test, n, emax * 1e3, std::sqrt(e2 / n) * 1e3, tol_mV, std::sqrt(rm2 / n) * 1e3, smax, resmax,
                (double)nit / n, 1e3 * wall / n, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    double tol = argc > 1 ? std::atof(argv[1]) : (DFN_DOUBLE ? 0.1 : 1.0);
    std::printf("open loop  %s  %s  (states %d, closure %d)\n", dfn_model::TAG, prec(), dfn::NZ, dfn::NG);
    int bad = 0;
    bad += run("HPPC20", tol);
    bad += run("CYC20", tol);
    return bad;
}
