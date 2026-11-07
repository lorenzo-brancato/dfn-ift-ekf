// Step Jacobian of the EKF (Lambda + U V^T, never formed on the board) against structjac.Struct.jac_full from Python,
// at the states of golden/jac.csv: rows z+ and V, columns z and the Rf coordinate (at Rf = 1).
#include <cmath>
#include "common.h"

static dfn::EKF ekf;
static double A[dfn::NZ + 1][dfn::NX];

int main() {
    Csv c;
    if (!read_csv(golden("jac.csv"), c)) { std::printf("no jac.csv\n"); return 0; }
    using namespace dfn;
    int bad = 0;
    double tol = DFN_DOUBLE ? 1e-6 : 3e-2;   // float: the Rf column (a sum that cancels to the total current) to ~1 %
    std::printf("Jacobian  %s  %s\n", dfn_model::TAG, prec());
    for (auto& r : c.rows) {
        const int k = (int)r[0];
        const double I = r[1];
        const double* z = &r[2];
        const double* gp = z + NZ;
        const double* Jz = gp + NG;
        const double* Jrf = Jz + (NZ + 1) * NZ;
        const int idt = dfn_model::EKF_IDT;                          // jac.csv is at the EKF step (export --ekf-dt)
        for (int i = 0; i < NZ; ++i) ekf.x[i] = (real_t)z[i];
        ekf.x[ITH] = 0;
        ekf.cl.load_abs(gp, ekf.g);
        EkfOut o;
        ekf.predict_lin((real_t)I, idt, o);
        double gmax = 0;
        for (int i = 0; i < NG; ++i) gmax = std::fmax(gmax, std::fabs(ekf.cl.abs_of(ekf.g, i) - gp[i]) / (1 + std::fabs(gp[i])));
        // dense A (unscaled) = Lambda + U V^T, rows z+ (NZ) and V
        for (int i = 0; i < NZ; ++i)
            for (int j = 0; j < NX; ++j) {
                double s = 0;
                for (int q = 0; q < NR; ++q) s += (double)ekf.Us[i][q] * (double)ekf.Vts[q][j];
                s *= (double)dfn_model::scale[i] / (double)dfn_model::scale[j];
                A[i][j] = s;
            }
        for (int i = 0; i < NZP; ++i) A[i][i] += ekf.dvec[i];
        for (int r2 = 0; r2 < N; ++r2)
            for (int l = 0; l < N; ++l) A[IE + r2][IE + l] += dfn_model::E[ekf.C.idt][r2][l];
        for (int j = 0; j < NX; ++j) A[NZ][j] = (double)ekf.hs[j] / (double)dfn_model::scale[j];
        // compare, relative to the largest entry of each row
        double emax = 0;
        int ei = -1, ej = -1;
        for (int i = 0; i <= NZ; ++i) {
            double rmax = 1e-300;
            for (int j = 0; j < NZ; ++j) rmax = std::fmax(rmax, std::fabs(Jz[i * NZ + j]));
            rmax = std::fmax(rmax, std::fabs(Jrf[i]));
            for (int j = 0; j <= NZ; ++j) {
                double ref = j < NZ ? Jz[i * NZ + j] : Jrf[i];
                double e = std::fabs(A[i][j] - ref) / rmax;
                if (e > emax) { emax = e; ei = i; ej = j; }
            }
        }
        bool ok = emax <= tol;
        bad += !ok;
        std::printf("  step %5d  I %+.3f A  |g - g_py| %.1e  max rel. error %.2e at (%d, %d)  %s\n", k, I, gmax, emax, ei, ej,
                    ok ? "PASS" : "FAIL");
    }
    return bad;
}
