// assadi_mbm.cpp — Baseline implementation of Algorithm 1 from
// S. Assadi, "A Simple (1-eps)-Approximation Semi-Streaming Algorithm for
// Maximum (Weighted) Matching" (arXiv:2307.02968), bipartite cardinality case.
//
// Faithful semi-streaming simulation:
//   * edges live only in a read-only "stream" (linear scans, pass-counted)
//   * NO per-edge persistent state: importances are implicit, recomputed from
//     per-vertex cover-history bitsets  q_e^(r) = 2^{c(e,r)}  (Lemma 3.3)
//   * per iteration: pass A -> Q^(r);  pass B -> sample;  solve on sample
//   * instrumentation: passes, peak sample size, quality trajectory
//
// Build: g++ -O2 -std=c++17 -o assadi_mbm assadi_mbm.cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge { int u, v; };            // u in [0,nL), v in [0,nR)

// ---------- Hopcroft–Karp with König minimum vertex cover ----------
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
        dist[u] = INF; return false;   // dead end
    }
    int maxMatching() {
        int res = 0;
        while (bfs())
            for (int u = 0; u < nL; ++u)
                if (matchL[u] < 0 && dfs(u)) ++res;
        return res;
    }
    // König: Z = alternating-reachable from unmatched L; cover = (L\Z) ∪ (R∩Z)
    void minVertexCover(vector<uint8_t>& coverL, vector<uint8_t>& coverR) {
        vector<uint8_t> visL(nL, 0), visR(nR, 0);
        queue<int> q;
        for (int u = 0; u < nL; ++u) if (matchL[u] < 0) { visL[u] = 1; q.push(u); }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (matchL[u] == v || visR[v]) continue;   // traverse non-matching edges L->R
                visR[v] = 1;
                int w = matchR[v];
                if (w >= 0 && !visL[w]) { visL[w] = 1; q.push(w); } // matching edge R->L
            }
        }
        coverL.assign(nL, 0); coverR.assign(nR, 0);
        for (int u = 0; u < nL; ++u) coverL[u] = !visL[u];
        for (int v = 0; v < nR; ++v) coverR[v] = visR[v];
    }
};

// ---------- per-vertex cover history as 64-bit words (decision point D2) ----------
struct CoverHistory {
    int n, words;                       // words = ceil(R/64)
    vector<uint64_t> bits;              // bits[v*words + w]
    CoverHistory(int n_, int R) : n(n_), words((R + 63) / 64), bits((size_t)n_ * words, 0) {}
    inline void set(int v, int r) { bits[(size_t)v * words + (r >> 6)] |= 1ULL << (r & 63); }
    // c(e,r) = # r' in [0,r) with both endpoints uncovered
    inline int uncoveredCount(int u, int v, const CoverHistory& other, int r) const {
        int full = r >> 6, rem = r & 63, c = 0;
        const uint64_t* A = &bits[(size_t)u * words];
        const uint64_t* B = &other.bits[(size_t)v * other.words];
        for (int w = 0; w < full; ++w) c += __builtin_popcountll(~(A[w] | B[w]));
        if (rem) c += __builtin_popcountll(~(A[full] | B[full]) & ((1ULL << rem) - 1));
        return c;
    }
};

struct RunStats {
    int iters = 0, passes = 0;
    size_t peakSample = 0;
    int best = 0;
    vector<int> trajectory;             // |M^(r)| per iteration
    double streamSec = 0, solveSec = 0;
};

// ---------- Algorithm 1, streaming-faithful ----------
RunStats runAlgorithm1(const vector<Edge>& stream, int nL, int nR, double eps,
                       uint64_t seed, int muExact /* validation only */,
                       int hardCapIters = 400) {
    RunStats S;
    const size_t m = stream.size();
    const int n = nL + nR;
    const int R = min<int>(hardCapIters, (int)ceil(4.0 / eps * log2((double)max<size_t>(m, 2))));
    S.passes = 1;                                     // pass 0: learn m (deviation log)
    CoverHistory HL(nL, R), HR(nR, R);
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    vector<Edge> sample;
    const int target = (int)ceil((1.0 - eps) * muExact);

    for (int r = 0; r < R; ++r) {
        auto t0 = chrono::steady_clock::now();
        // ---- pass A: Q^(r) = sum_e 2^{c(e,r)} ----
        ++S.passes;
        double Q = 0;
        for (const Edge& e : stream) {
            int c = HL.uncoveredCount(e.u, e.v, HR, r);
            Q += ldexp(1.0, c);                        // D1: double baseline, guarded
        }
        assert(isfinite(Q));
        // ---- pass B: sample with p_e = min(1, (2n/eps) * q_e / Q) ----
        ++S.passes;
        sample.clear();
        const double scale = (2.0 * n / eps) / Q;
        for (const Edge& e : stream) {
            int c = HL.uncoveredCount(e.u, e.v, HR, r);
            double p = min(1.0, ldexp(scale, c));
            if (U(rng) < p) sample.push_back(e);
        }
        S.peakSample = max(S.peakSample, sample.size());
        auto t1 = chrono::steady_clock::now();
        // ---- solve on the sample (offline, allowed) ----
        HopcroftKarp hk(nL, nR);
        for (const Edge& e : sample) hk.addEdge(e.u, e.v);
        int M = hk.maxMatching();
        vector<uint8_t> cL, cR;
        hk.minVertexCover(cL, cR);
        for (int u = 0; u < nL; ++u) if (cL[u]) HL.set(u, r);
        for (int v = 0; v < nR; ++v) if (cR[v]) HR.set(v, r);
        auto t2 = chrono::steady_clock::now();
        S.streamSec += chrono::duration<double>(t1 - t0).count();
        S.solveSec  += chrono::duration<double>(t2 - t1).count();
        S.best = max(S.best, M);
        S.trajectory.push_back(M);
        S.iters = r + 1;
        if (S.best >= target) break;                   // validation-only stop (D4)
    }
    return S;
}

// ---------- instance generators ----------
vector<Edge> genER(int nL, int nR, double avgDegL, mt19937_64& rng) {
    double p = avgDegL / nR;
    vector<Edge> es; bernoulli_distribution B(p);
    for (int u = 0; u < nL; ++u)
        for (int v = 0; v < nR; ++v)
            if (B(rng)) es.push_back({u, v});
    shuffle(es.begin(), es.end(), rng);               // adversarial-ish arrival order
    return es;
}
// hidden perfect matching + dense noise concentrated on few vertices:
// stresses the importance mechanism (matching edges must get boosted)
vector<Edge> genHidden(int n, int noiseHubs, int hubDeg, mt19937_64& rng) {
    vector<int> perm(n); iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);
    vector<Edge> es;
    for (int u = 0; u < n; ++u) es.push_back({u, perm[u]});   // hidden PM
    uniform_int_distribution<int> DV(0, n - 1);
    for (int h = 0; h < noiseHubs; ++h) {
        int hub = DV(rng);
        for (int k = 0; k < hubDeg; ++k) es.push_back({hub, DV(rng)});
    }
    sort(es.begin(), es.end(), [](const Edge&a, const Edge&b){return a.u<b.u||(a.u==b.u&&a.v<b.v);});
    es.erase(unique(es.begin(), es.end(), [](const Edge&a,const Edge&b){return a.u==b.u&&a.v==b.v;}), es.end());
    shuffle(es.begin(), es.end(), rng);
    return es;
}

int exactMu(const vector<Edge>& es, int nL, int nR) {
    HopcroftKarp hk(nL, nR);
    for (const Edge& e : es) hk.addEdge(e.u, e.v);
    return hk.maxMatching();
}

int main(int argc, char** argv) {
    uint64_t seed = argc > 1 ? strtoull(argv[1], nullptr, 10) : 20260707ULL;
    mt19937_64 rng(seed);
    printf("%-28s %8s %6s %6s %5s %6s %6s %9s %7s %8s %8s\n",
           "instance", "m", "mu", "eps", "R", "iters", "passes",
           "peakSamp", "best", "ratio", "solve_s");
    struct Case { string name; vector<Edge> es; int nL, nR; };
    vector<Case> cases;
    // regime requirement: avg degree >> 4/eps, i.e., m >> 2n/eps (else p_e caps at 1
    // and the whole stream is sampled in iteration 1 — degenerate, see results log)
    cases.push_back({"ER n=2000 d=120",  genER(2000, 2000, 120, rng), 2000, 2000});
    cases.push_back({"ER n=2000 d=400",  genER(2000, 2000, 400, rng), 2000, 2000});
    cases.push_back({"ER n=4000 d=250",  genER(4000, 4000, 250, rng), 4000, 4000});
    cases.push_back({"hiddenPM n=2000 hubs=400", genHidden(2000, 400, 4000, rng), 2000, 2000});
    for (double eps : {0.1, 0.05}) {
        for (auto& C : cases) {
            int mu = exactMu(C.es, C.nL, C.nR);
            int Rwc = (int)ceil(4.0 / eps * log2((double)C.es.size()));
            RunStats S = runAlgorithm1(C.es, C.nL, C.nR, eps, seed ^ hash<string>{}(C.name), mu);
            printf("%-28s %8zu %6d %6.2f %5d %6d %6d %9zu %7d %8.4f %8.2f\n",
                   C.name.c_str(), C.es.size(), mu, eps, Rwc, S.iters, S.passes,
                   S.peakSample, S.best, (double)S.best / mu, S.solveSec);
        }
    }
    return 0;
}
