// phase4.cpp — Adaptive boosting (Theorem B) + deterministic certificate (Theorem A),
// head-to-head against the fixed-2 rule on the Phase-2 benchmark suite.
//
// Importances in log2-space: lq_e^(r) = sum over {r' < r : e uncovered at r'} of log2(beta_{r'}).
// Adaptive iteration = 3 passes: A (Q^(r)), B (sample), C (uncMass_r with U^(r) -> rho_r -> beta_r).
// Fixed-2 iteration = 3 passes here as well (same accounting, apples-to-apples certificate).
// Certificate: Phi += eps*log2(beta_r) - log2(1+(beta_r-1)*rho_r); fires when Phi > log2(m).
// rho_r == 0  =>  U^(r) is a vertex cover of G  =>  M^(r) certified maximum.
// Build: g++ -O2 -std=c++17 -o phase4 phase4.cpp
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
};

// log2 q_e^(r) = sum of lb[r'] over r' < r with both endpoints uncovered
static inline double logq(const CoverHistory& HL, int u, const CoverHistory& HR, int v,
                          int r, const vector<double>& lb) {
    double s = 0;
    int full = r >> 6, rem = r & 63;
    const uint64_t* A = &HL.bits[(size_t)u * HL.words];
    const uint64_t* B = &HR.bits[(size_t)v * HR.words];
    for (int w = 0; w <= full; ++w) {
        uint64_t un = ~(A[w] | B[w]);
        if (w == full) { if (!rem) break; un &= (1ULL << rem) - 1; }
        while (un) { int b = __builtin_ctzll(un); s += lb[(w << 6) + b]; un &= un - 1; }
    }
    return s;
}

struct IterLog { int r, M; size_t samp; double rho, beta, Phi; };
struct RunResult {
    int iters = 0, passes = 0, best = 0;
    int qualIter = -1;          // first iteration reaching (1-eps)*mu (validation)
    int certIter = -1;          // iteration where deterministic certificate fires
    size_t peakSample = 0;
    vector<IterLog> log;
};

RunResult run(const vector<Edge>& stream, int nL, int nR, double eps, bool adaptive,
              uint64_t seed, int mu, int maxIters, double Bcap = 1024.0) {
    RunResult S;
    const size_t m = stream.size();
    const int n = nL + nR;
    const double lm = log2((double)m);
    CoverHistory HL(nL, maxIters), HR(nR, maxIters);
    vector<double> lb;                       // log2(beta_r), grows per iteration
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    vector<Edge> sample;
    const int target = (int)ceil((1.0 - eps) * mu);
    double Phi = 0;
    S.passes = 1;                            // pass 0: learn m

    for (int r = 0; r < maxIters; ++r) {
        // pass A: Q^(r)
        ++S.passes;
        double Q = 0;
        for (const Edge& e : stream) Q += exp2(logq(HL, e.u, HR, e.v, r, lb));
        assert(isfinite(Q));
        // pass B: sample
        ++S.passes;
        sample.clear();
        const double lscale = log2(2.0 * n / eps) - log2(Q);
        for (const Edge& e : stream) {
            double lp = lscale + logq(HL, e.u, HR, e.v, r, lb);
            if (lp >= 0 || U(rng) < exp2(lp)) sample.push_back(e);
        }
        S.peakSample = max(S.peakSample, sample.size());
        // solve
        HopcroftKarp hk(nL, nR);
        for (const Edge& e : sample) hk.addEdge(e.u, e.v);
        int M = hk.maxMatching();
        vector<uint8_t> cL, cR;
        hk.minVertexCover(cL, cR);
        for (int u = 0; u < nL; ++u) if (cL[u]) HL.set(u, r);
        for (int v = 0; v < nR; ++v) if (cR[v]) HR.set(v, r);
        S.best = max(S.best, M);
        if (S.qualIter < 0 && S.best >= target) S.qualIter = r + 1;
        // pass C: uncMass_r with q^(r) and U^(r)
        ++S.passes;
        double uncMass = 0;
        for (const Edge& e : stream)
            if (!HL.get(e.u, r) && !HR.get(e.v, r))
                uncMass += exp2(logq(HL, e.u, HR, e.v, r, lb));
        double rho = uncMass / Q;
        // choose beta_r
        double beta;
        if (rho <= 0) { S.certIter = r + 1; S.best = max(S.best, M);   // U^(r) covers G: M maximum
                        S.log.push_back({r + 1, M, sample.size(), rho, 0.0, Phi});
                        S.iters = r + 1; S.qualIter = S.qualIter < 0 ? r + 1 : S.qualIter; break; }
        if (adaptive) beta = min(Bcap, max(2.0, eps * (1 - rho) / (rho * (1 - eps))));
        else          beta = 2.0;
        lb.push_back(log2(beta));
        Phi += eps * log2(beta) - log2(1 + (beta - 1) * rho);
        S.log.push_back({r + 1, M, sample.size(), rho, beta, Phi});
        S.iters = r + 1;
        if (S.certIter < 0 && Phi > lm) S.certIter = r + 1;
        if (S.qualIter > 0 && S.certIter > 0) break;      // both milestones observed
    }
    return S;
}

// ---- generators (as Phase 2) ----
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
vector<Edge> genRMAT(int scale, size_t mTarget, mt19937_64& rng) {
    const double a = 0.57, b = 0.19, c = 0.19;
    uniform_real_distribution<double> U(0, 1);
    set<pair<int,int>> seen; vector<Edge> es;
    while (es.size() < mTarget) {
        int u = 0, v = 0;
        for (int s = 0; s < scale; ++s) {
            double x = U(rng);
            u = 2 * u + ((x < a + b) ? 0 : 1);
            v = 2 * v + ((x < a || (x >= a + b && x < a + b + c)) ? 0 : 1);
        }
        if (seen.insert({u, v}).second) es.push_back({u, v});
    }
    shuffle(es.begin(), es.end(), rng);
    return es;
}
vector<Edge> genER(int nL, int nR, double avgDegL, mt19937_64& rng) {
    double p = avgDegL / nR;
    vector<Edge> es; bernoulli_distribution B(p);
    for (int u = 0; u < nL; ++u)
        for (int v = 0; v < nR; ++v)
            if (B(rng)) es.push_back({u, v});
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
    struct Case { string name; vector<Edge> es; int nL, nR; };
    vector<Case> cases;
    cases.push_back({"hiddenPM n=2000", genHidden(2000, 400, 4000, rng), 2000, 2000});
    cases.push_back({"hiddenPM n=4000", genHidden(4000, 800, 4000, rng), 4000, 4000});
    cases.push_back({"RMAT-12 m=1M",    genRMAT(12, 1000000, rng), 4096, 4096});
    cases.push_back({"ER n=2000 d=400", genER(2000, 2000, 400, rng), 2000, 2000});

    printf("%-18s %5s | %-8s | %6s %7s %7s %8s %8s | %s\n",
           "instance", "eps", "rule", "qualIt", "qualPas", "certIt", "best/mu", "peakSamp",
           "beta trace (first 8)");
    for (auto& C : cases) {
        int mu = exactMu(C.es, C.nL, C.nR);
        for (double eps : {0.1, 0.05}) {
            for (int ad = 0; ad <= 1; ++ad) {
                RunResult S = run(C.es, C.nL, C.nR, eps, ad,
                                  seed ^ hash<string>{}(C.name) ^ (uint64_t)(eps * 1000),
                                  mu, 500);
                string bt;
                for (size_t i = 0; i < min<size_t>(8, S.log.size()); ++i) {
                    char b[32]; snprintf(b, 32, "%.1f ", S.log[i].beta); bt += b;
                }
                int qualPasses = S.qualIter > 0 ? 1 + 3 * S.qualIter : -1;
                printf("%-18s %5.2f | %-8s | %6d %7d %7d %8.4f %8zu | %s\n",
                       C.name.c_str(), eps, ad ? "adaptive" : "fixed-2",
                       S.qualIter, qualPasses, S.certIter, (double)S.best / mu,
                       S.peakSample, bt.c_str());
                string csv = "p4_" + string(ad ? "adapt" : "fixed") + "_" +
                             C.name.substr(0, C.name.find(' ')) + "_n" + to_string(C.nL) +
                             "_eps" + to_string((int)round(eps * 100)) + ".csv";
                FILE* f = fopen(csv.c_str(), "w");
                fprintf(f, "r,M,ratio,rho,beta,Phi\n");
                for (auto& L : S.log)
                    fprintf(f, "%d,%d,%.6f,%.6g,%.3f,%.4f\n",
                            L.r, L.M, (double)L.M / mu, L.rho, L.beta, L.Phi);
                fclose(f);
            }
        }
    }
    return 0;
}
