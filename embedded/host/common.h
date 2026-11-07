// Host-side helpers of the tests: CSV reading, timing, the golden folder of the compiled model.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dfn/ekf.h"

#ifndef DFN_GENERATED_DIR
#error "DFN_GENERATED_DIR must name embedded/generated"
#endif

struct Csv {
    std::vector<std::string> names;
    std::vector<std::vector<double>> rows;
    int col(const std::string& n) const {
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == n) return (int)i;
        std::fprintf(stderr, "column %s not found\n", n.c_str());
        std::exit(2);
    }
    std::vector<double> get(const std::string& n) const {
        int c = col(n);
        std::vector<double> v(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) v[i] = rows[i][c];
        return v;
    }
};

inline bool read_csv(const std::string& path, Csv& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::getline(f, line);
    {
        std::stringstream ss(line); std::string s;
        while (std::getline(ss, s, ',')) out.names.push_back(s);
    }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> r;
        r.reserve(out.names.size());
        const char* p = line.c_str();
        char* e;
        while (*p) {
            r.push_back(std::strtod(p, &e));
            p = e;
            if (*p == ',') ++p; else break;
        }
        out.rows.push_back(std::move(r));
    }
    return true;
}

inline std::string golden(const std::string& file) {
    return std::string(DFN_GENERATED_DIR) + "/" + dfn_model::TAG + "/golden/" + file;
}

inline double read_scalar(const std::string& path) {
    std::ifstream f(path);
    double v = 0;
    f >> v;
    return v;
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double s() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

inline const char* prec() {
    return DFN_DOUBLE ? (DFN_COV_UD ? "double, UD" : "double, Joseph")
                      : (DFN_CLOSURE_DOUBLE ? "float (closure double), UD" : (DFN_COV_UD ? "float, UD" : "float, Joseph"));
}

// state with q [A.h] below the top of the window (rested): z0 is at dfn_model::q0, z is linear in q
inline void z_at(double q, real_t* z) {
    for (int i = 0; i < dfn::NZ; ++i) z[i] = (real_t)(dfn_model::z0[i] + (q - dfn_model::q0) * dfn_model::dz_dq[i]);
}

// smallest pivot of a Cholesky factorization of a copy of P (negative or NaN: not positive definite)
inline double chol_min(const double (*P)[dfn::NX]) {
    static double L[dfn::NX][dfn::NX];
    double mn = 1e300;
    for (int j = 0; j < dfn::NX; ++j) {
        double s = P[j][j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
        if (!(s > 0)) return s;
        double d = std::sqrt(s);
        if (d < mn) mn = d;
        L[j][j] = d;
        for (int i = j + 1; i < dfn::NX; ++i) {
            double t = P[i][j];
            for (int k = 0; k < j; ++k) t -= L[i][k] * L[j][k];
            L[i][j] = t / d;
        }
    }
    return mn * mn;
}
