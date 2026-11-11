// The board's start, on the host: the EKF initialized from the OCV as xiao_online's start_from_ocv (q at the measured
// voltage, by the same grid search) at a given temperature, then EKF steps at rest. Reproduces a board that stops at
// its first EKF step.
//   board_start_<tag> [V] [T_degC] [steps] [I_A]
#include <cmath>
#include "common.h"

static dfn::EKF ekf;

static double q_from_ocv(double V) {                         // xiao_online/src/main.cpp
    using namespace dfn_model;
    double best = 0, eb = 1e9;
    for (int i = 0; i <= 5600; ++i) {
        double q = -0.4 + i * 0.0005;
        double x = (double)x100 - q / (double)Q_n, y = (double)y100 + q / (double)Q_p;
        if (x <= 0.001 || x >= 0.999 || y <= 0.001 || y >= 0.999) continue;
        double U = dfn::Up<double>(y) - dfn::Un<double>(x)
                 + (double)dfn::TH.entdT * ((double)ent_d0 + (double)ent_dn * (x - 0.5) + (double)ent_dp * (y - 0.5));
        if (U == U && std::fabs(U - V) < eb) { eb = std::fabs(U - V); best = q; }
    }
    return best;
}

int main(int argc, char** argv) {
    const double V = argc > 1 ? std::atof(argv[1]) : 4.148, Tc = argc > 2 ? std::atof(argv[2]) : 30.0;
    const int n = argc > 3 ? std::atoi(argv[3]) : 10;
    const double I = argc > 4 ? std::atof(argv[4]) : 0.0;
    std::printf("board start  %s  %s  V %.4f  T %.2f degC  I %.4f A\n", dfn_model::TAG, prec(), V, Tc, I);
    dfn::TH.set((real_t)Tc);
    const double q = q_from_ocv(V);
    real_t z0[dfn::NZ];
    z_at(q, z0);
    ekf.init(z0, (real_t)0, dfn_model::EKF_IDT, 0);
    std::printf("  q %.4f A.h  SoC %.3f %%  window y %.4f..%.4f\n", q, dfn::soc_of_ybar(dfn::ybar(ekf.x)) * 100,
                (double)dfn::SW.y_top, (double)dfn::SW.y_bot);
    for (int k = 0; k < n; ++k) {
        dfn::TH.set((real_t)Tc);
        Timer t;
        dfn::EkfOut o = ekf.step((real_t)I, (real_t)V, dfn_model::EKF_IDT, true);
        std::printf("  step %2d: SoC %.4f sd %.5f  V_est %.5f innov %+.6f  Rf %.4f LLI %.5f  res %.2e nit %d  %.1f ms\n", k,
                    (double)o.soc, (double)o.sd_soc, (double)o.V, (double)o.innov, (double)o.Rf, (double)o.LLI,
                    (double)o.res, o.nit, t.s() * 1e3);
        if (!(o.soc == o.soc)) { std::printf("  state not finite\n"); return 1; }
    }
    return 0;
}
