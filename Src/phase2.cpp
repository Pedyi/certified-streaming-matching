// phase2.cpp — Experiment harness for the engineering study.
// Adds to the Phase-1 baseline:
//   * MatrixMarket loader (rows = L, cols = R; symmetric expansion; diag dropped)
//   * RMAT and Chung–Lu bipartite generators (realistic skewed degrees)
//   * Instrumentation: per-iteration CSV with |M^(r)|, sample size, Q^(r), and
//     rho^(r-1) = uncovered-mass ratio, computed at ZERO extra passes by folding
//     into pass A of the next iteration:
//        c(e,r) known  =>  c(e,r-1) = c(e,r) - [e uncovered at r-1]
//        q_e^(r-1) = 2^{c(e,r-1)};  accumulate over uncovered edges.
//     (Lemma 3.1 guarantees rho <= eps/2 w.e.h.p. — we test this empirically.)
// Build: g++ -O2 -std=c++17 -o phase2 phase2.cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge { int u, v; };

struct HopcroftKarp {
    int nL, nR;
    vector<vector<int>> adj;
    vector<int> matchL, matchR, dist;
    static constexpr int INF = INT_MAX;
    HopcroftKarp(int nL_, int nR_)
        : nL(nL_), nR(nR_), adj(nL_), matchL(nL_, -1), matchR(nR_, -1) {}
    void addEdge(int u, int v) { adj[u].push_back(v); }
    bool bfs() {
        queue<int> q; dist.assign(nL, INF); bool found = false;
        for (int u = 0; u < nL; ++u) if (matchL[u] < 0) { dist[u] = 0; q.push(u); }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                int w = matchR[v];
                if (w < 0) found = true;
                else if (dist[w] == INF) { dist[w] = dist[u] + 1; q.push(w); }
            }
        }
        return found;
    }
    bool dfs(int u) {
        for (int v : adj[u]) {
            int w = matchR[v];
            if (w < 0 || (dist[w] == dist[u] + 1 && dfs(w))) {
                matchL[u] = v; matchR[v] = u; return true;
            }
        }
        dist[u] = INF; return false;
    }
    int maxMatching() {
        int res = 0;
        while (bfs())
            for (int u = 0; u < nL; ++u)
                if (matchL[u] < 0 && dfs(u)) ++res;
        return res;
    }
    void minVertexCover(vector<uint8_t>& coverL, vector<uint8_t>& coverR) {
        vector<uint8_t> visL(nL, 0), visR(nR, 0);
        queue<int> q;
        for (int u = 0; u < nL; ++u) if (matchL[u] < 0) { visL[u] = 1; q.push(u); }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (matchL[u] == v || visR[v]) continue;
                visR[v] = 1;
                int w = matchR[v];
                if (w >= 0 && !visL[w]) { visL[w] = 1; q.push(w); }
            }
        }
        coverL.assign(nL, 0); coverR.assign(nR, 0);
        for (int u = 0; u < nL; ++u) coverL[u] = !visL[u];
        for (int v = 0; v < nR; ++v) coverR[v] = visR[v];
    }
};

struct CoverHistory {
    int n, words;
    vector<uint64_t> bits;
    CoverHistory(int n_, int R) : n(n_), words((R + 63) / 64), bits((size_t)n_ * words, 0) {}
    inline void set(int v, int r) { bits[(size_t)v * words + (r >> 6)] |= 1ULL << (r & 63); }
    inline bool get(int v, int r) const { return bits[(size_t)v * words + (r >> 6)] >> (r & 63) & 1; }
    inline int uncoveredCount(int u, const CoverHistory& o, int v, int r) const {
        int full = r >> 6, rem = r & 63, c = 0;
        const uint64_t* A = &bits[(size_t)u * words];
        const uint64_t* B = &o.bits[(size_t)v * o.words];
        for (int w = 0; w < full; ++w) c += __builtin_popcountll(~(A[w] | B[w]));
        if (rem) c += __builtin_popcountll(~(A[full] | B[full]) & ((1ULL << rem) - 1));
        return c;
    }
};

struct IterLog { int r, M; size_t samp; double Q, rho; };
struct RunResult {
    int iters = 0, passes = 0, best = 0, mu = 0;
    size_t peakSample = 0;
    vector<IterLog> log;
};

RunResult runInstrumented(const vector<Edge>& stream, int nL, int nR, double eps,
                          uint64_t seed, int mu, int hardCap = 400) {
    RunResult S; S.mu = mu;
    const size_t m = stream.size();
    const int n = nL + nR;
    const int R = min<int>(hardCap, (int)ceil(4.0 / eps * log2((double)max<size_t>(m, 2))));
    S.passes = 1;                                        // pass 0: learn m
    CoverHistory HL(nL, R), HR(nR, R);
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    vector<Edge> sample;
    const int target = (int)ceil((1.0 - eps) * mu);
    double prevRho = NAN;

    for (int r = 0; r < R; ++r) {
        // ---- pass A: Q^(r), and rho^(r-1) folded in for free ----
        ++S.passes;
        double Q = 0, uncMassPrev = 0, Qprev = 0;
        for (const Edge& e : stream) {
            int c = HL.uncoveredCount(e.u, HR, e.v, r);
            Q += ldexp(1.0, c);
            if (r > 0) {
                bool unc = !HL.get(e.u, r - 1) && !HR.get(e.v, r - 1);
                double qp = ldexp(1.0, c - (unc ? 1 : 0));   // q_e^(r-1)
                Qprev += qp;
                if (unc) uncMassPrev += qp;
            }
        }
        assert(isfinite(Q));
        if (r > 0) prevRho = uncMassPrev / Qprev, S.log.back().rho = prevRho;
        // ---- pass B: sample ----
        ++S.passes;
        sample.clear();
        const double scale = (2.0 * n / eps) / Q;
        for (const Edge& e : stream) {
            int c = HL.uncoveredCount(e.u, HR, e.v, r);
            if (U(rng) < min(1.0, ldexp(scale, c))) sample.push_back(e);
        }
        S.peakSample = max(S.peakSample, sample.size());
        // ---- solve ----
        HopcroftKarp hk(nL, nR);
        for (const Edge& e : sample) hk.addEdge(e.u, e.v);
        int M = hk.maxMatching();
        vector<uint8_t> cL, cR;
        hk.minVertexCover(cL, cR);
        for (int u = 0; u < nL; ++u) if (cL[u]) HL.set(u, r);
        for (int v = 0; v < nR; ++v) if (cR[v]) HR.set(v, r);
        S.best = max(S.best, M);
        S.log.push_back({r + 1, M, sample.size(), Q, NAN});
        S.iters = r + 1;
        if (S.best >= target) break;                     // validation-only stop
    }
    return S;
}

// ---------------- loaders / generators ----------------
bool loadMTX(const string& path, vector<Edge>& es, int& nL, int& nR) {
    ifstream f(path); if (!f) return false;
    string line; bool symmetric = false;
    getline(f, line);
    if (line.find("MatrixMarket") == string::npos) return false;
    symmetric = line.find("symmetric") != string::npos;
    while (getline(f, line) && !line.empty() && line[0] == '%') {}
    size_t nnz; { istringstream ss(line); ss >> nL >> nR >> nnz; }
    es.clear(); es.reserve(symmetric ? 2 * nnz : nnz);
    int i, j; double val;
    while (getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        istringstream ss(line); ss >> i >> j; --i; --j;
        if (i == j) continue;                    // diagonal irrelevant for matching
        es.push_back({i, j});
        if (symmetric) es.push_back({j, i});
    }
    return true;
}
vector<Edge> genRMAT(int scaleL, int scaleR, size_t mTarget, mt19937_64& rng) {
    // Graph500 parameters, bipartite grid 2^scaleL x 2^scaleR
    const double a = 0.57, b = 0.19, c = 0.19; // d = 0.05
    uniform_real_distribution<double> U(0, 1);
    set<pair<int,int>> seen;
    vector<Edge> es;
    while (es.size() < mTarget) {
        int u = 0, v = 0;
        for (int s = 0; s < max(scaleL, scaleR); ++s) {
            double x = U(rng);
            int qu = (x < a + b) ? 0 : 1;              // row half
            int qv = (x < a || (x >= a + b && x < a + b + c)) ? 0 : 1;
            if (s < scaleL) u = 2 * u + qu;
            if (s < scaleR) v = 2 * v + qv;
        }
        if (seen.insert({u, v}).second) es.push_back({u, v});
    }
    shuffle(es.begin(), es.end(), rng);
    return es;
}
vector<Edge> genChungLu(int nL, int nR, size_t mTarget, double beta, mt19937_64& rng) {
    vector<double> wL(nL), wR(nR);
    for (int i = 0; i < nL; ++i) wL[i] = pow(i + 1, -beta);
    for (int i = 0; i < nR; ++i) wR[i] = pow(i + 1, -beta);
    double SL = accumulate(wL.begin(), wL.end(), 0.0);
    discrete_distribution<int> DR(wR.begin(), wR.end());
    set<pair<int,int>> seen; vector<Edge> es;
    // expected degree of u proportional to wL[u]
    vector<double> cum(nL); partial_sum(wL.begin(), wL.end(), cum.begin());
    uniform_real_distribution<double> U(0, SL);
    while (es.size() < mTarget) {
        int u = lower_bound(cum.begin(), cum.end(), U(rng)) - cum.begin();
        int v = DR(rng);
        if (seen.insert({u, v}).second) es.push_back({u, v});
    }
    shuffle(es.begin(), es.end(), rng);
    return es;
}
vector<Edge> genHidden(int n, int hubs, int hubDeg, mt19937_64& rng) {
    vector<int> perm(n); iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);
    set<pair<int,int>> seen; vector<Edge> es;
    for (int u = 0; u < n; ++u) { seen.insert({u, perm[u]}); es.push_back({u, perm[u]}); }
    uniform_int_distribution<int> DV(0, n - 1);
    for (int h = 0; h < hubs; ++h) {
        int hub = DV(rng);
        for (int k = 0; k < hubDeg; ++k) {
            int v = DV(rng);
            if (seen.insert({hub, v}).second) es.push_back({hub, v});
        }
    }
    shuffle(es.begin(), es.end(), rng);
    return es;
}
int exactMu(const vector<Edge>& es, int nL, int nR) {
    HopcroftKarp hk(nL, nR);
    for (const Edge& e : es) hk.addEdge(e.u, e.v);
    return hk.maxMatching();
}

int main() {
    uint64_t seed = 20260707ULL;
    mt19937_64 rng(seed);
    struct Case { string name; vector<Edge> es; int nL, nR; vector<double> epss; };
    vector<Case> cases;
    { vector<Edge> es; int nL, nR;
      if (loadMTX("bcsstk13.mtx", es, nL, nR))
          cases.push_back({"bcsstk13(real)", es, nL, nR, {0.2, 0.1}});
      else fprintf(stderr, "bcsstk13.mtx not found — skipping real instance\n"); }
    cases.push_back({"RMAT-12", genRMAT(12, 12, 1000000, rng), 4096, 4096, {0.1, 0.05}});
    cases.push_back({"ChungLu-12 b=0.5", genChungLu(4096, 4096, 800000, 0.5, rng), 4096, 4096, {0.1, 0.05}});
    cases.push_back({"hiddenPM n=2000", genHidden(2000, 400, 4000, rng), 2000, 2000, {0.1, 0.05}});

    FILE* summ = fopen("phase2_summary.txt", "w");
    fprintf(summ, "%-22s %8s %6s %6s %6s %6s %6s %9s %8s %8s\n",
            "instance", "m", "mu", "eps", "R_wc", "iters", "passes", "peakSamp", "best", "ratio");
    for (auto& C : cases) {
        int mu = exactMu(C.es, C.nL, C.nR);
        for (double eps : C.epss) {
            int Rwc = (int)ceil(4.0 / eps * log2((double)C.es.size()));
            RunResult S = runInstrumented(C.es, C.nL, C.nR, eps,
                                          seed ^ hash<string>{}(C.name) ^ (uint64_t)(eps*1000),
                                          mu);
            fprintf(summ, "%-22s %8zu %6d %6.2f %6d %6d %6d %9zu %8d %8.4f\n",
                    C.name.c_str(), C.es.size(), mu, eps, Rwc, S.iters, S.passes,
                    S.peakSample, S.best, (double)S.best / mu);
            string csv = "traj_" + C.name.substr(0, C.name.find_first_of(" (")) +
                         "_eps" + to_string((int)round(eps * 100)) + ".csv";
            FILE* f = fopen(csv.c_str(), "w");
            fprintf(f, "r,M,ratio,sample,Q,rho\n");
            for (auto& L : S.log)
                fprintf(f, "%d,%d,%.6f,%zu,%.6g,%.6g\n",
                        L.r, L.M, (double)L.M / mu, L.samp, L.Q, L.rho);
            fclose(f);
        }
    }
    fclose(summ);
    // echo summary
    ifstream f("phase2_summary.txt"); string l;
    while (getline(f, l)) puts(l.c_str());
    return 0;
}
