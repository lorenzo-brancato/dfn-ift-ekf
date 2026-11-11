// Temperature tables (dfn::TH.set) against Python models built at other temperatures: the CYC20 open loop of
// golden/openloop_CYC20_T<T>.csv. Grid temperatures test the tables, others the interpolation between grid points.
#include <cmath>
#include <cstring>
#include "common.h"

static dfn::OpenLoop ol;

int main(int argc, char** argv) {
    double tol = argc > 1 ? std::atof(argv[1]) : 2.0;         // mV (between grid points: interpolation error)
    std::printf("temperature  %s  %s  tables: %s\n", dfn_model::TAG, prec(), dfn::TH.tables() ? "yes" : "no");
    if (!dfn::TH.tables()) return 0;
    const double Ts[] = {0.0, 27.5, 40.0, 20.0};
    int bad = 0;
    for (double Tc : Ts) {
        char name[64];
        std::snprintf(name, sizeof name, "openloop_CYC20_T%g.csv", Tc);
        Csv c;
        if (!read_csv(golden(name), c)) {
            if (Tc != 20.0) std::printf("  %5.1f degC: no golden file, skipped\n", Tc);
            if (Tc != 20.0) continue;
            if (!read_csv(golden("openloop_CYC20.csv"), c)) continue;
        }
        dfn::TH.reset();
        dfn::TH.set((real_t)Tc);
        std::vector<double> I = c.get("I"), Vp = c.get("V"), idt = c.get("idt");
        real_t z0[dfn::NZ];
        z_at(dfn_model::q0, z0);
        ol.reset(z0);
        double emax = 0, e2 = 0;
        size_t n = I.size();                                  // up to the first implausible Python voltage
        for (size_t k = 0; k < I.size(); ++k)                 // (at 0 degC a 1C discharge collapses near 23 % SOC)
            if (!(Vp[k] >= 2.5 && Vp[k] <= 4.4)) { n = k; break; }
        I.resize(n);
        for (size_t k = 0; k < n; ++k) {
            double V = ol.step((real_t)I[k], (int)idt[k]);
            double e = std::fabs(V - Vp[k]);
            if (!(e <= emax)) emax = e;
            e2 += e * e;
        }
        bool on_grid = std::fmod(Tc + 100.0, 5.0) == 0.0;
        double t = on_grid ? (DFN_DOUBLE ? 0.1 : 1.0) : tol;
        bool ok = emax * 1e3 <= t;
        bad += !ok;
        std::printf("  %5.1f degC (%s): %zu steps  |V - V_py| max %.4f mV rms %.4f mV (tol %.2f)  %s\n", Tc,
                    on_grid ? "grid point" : "interpolated", I.size(), emax * 1e3, std::sqrt(e2 / I.size()) * 1e3, t,
                    ok ? "PASS" : "FAIL");
    }
    dfn::TH.reset();
    return bad;
}
