// robustness.cpp — Multi-seed robustness study: fixed-2 vs adaptive boosting.
// Self-contained. Usage:
//   g++ -O3 -march=native -std=c++17 -o robustness robustness.cpp
//   ./robustness [--seeds N] [--maxit N] [file1.mtx file2.mtx ...]
// Defaults: --seeds 10 --maxit 500.
// Design: each named instance is generated ONCE with a fixed generator seed;
// variance reported is over ALGORITHM randomness only (clean experimental design).
// Outputs:
//   robustness_raw.csv      one row per (instance,eps,rule,seed)
//   stdout                  aggregate table: mean +/- sd of key metrics
// Metrics: qualIter (first iter reaching (1-eps)*mu), qualPasses = 1+3*qualIter,
//          certIter (Theorem-A certificate fires; type R = rho=0 exact-optimality,
//          type P = Phi > log2 m), peakSamp, ratio (best/mu at stop).
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

struct RunResult {
    int qualIter = -1, certIter = -1;
    char certType = '-';                   // 'R' rho=0, 'P' Phi>log2 m
    size_t peakSample = 0;
    double ratio = 0;
};

RunResult run(const vector<Edge>& stream, int nL, int nR, double eps, bool adaptive,
              uint64_t seed, int mu, int maxIters, double Bcap = 1024.0) {
    RunResult S;
    const size_t m = stream.size();
    const int n = nL + nR;
    const double lm = log2((double)m);
    CoverHistory HL(nL, maxIters), HR(nR, maxIters);
    vector<double> lb;
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    vector<Edge> sample;
    const int target = (int)ceil((1.0 - eps) * mu);
    double Phi = 0;
    int best = 0;

    for (int r = 0; r < maxIters; ++r) {
        double Q = 0;
        for (const Edge& e : stream) Q += exp2(logq(HL, e.u, HR, e.v, r, lb));
        assert(isfinite(Q));
        sample.clear();
        const double lscale = log2(2.0 * n / eps) - log2(Q);
        for (const Edge& e : stream) {
            double lp = lscale + logq(HL, e.u, HR, e.v, r, lb);
            if (lp >= 0 || U(rng) < exp2(lp)) sample.push_back(e);
        }
        S.peakSample = max(S.peakSample, sample.size());
        HopcroftKarp hk(nL, nR);
        for (const Edge& e : sample) hk.addEdge(e.u, e.v);
        int M = hk.maxMatching();
        vector<uint8_t> cL, cR;
        hk.minVertexCover(cL, cR);
        for (int u = 0; u < nL; ++u) if (cL[u]) HL.set(u, r);
        for (int v = 0; v < nR; ++v) if (cR[v]) HR.set(v, r);
        best = max(best, M);
        if (S.qualIter < 0 && best >= target) S.qualIter = r + 1;
        double uncMass = 0;
        for (const Edge& e : stream)
            if (!HL.get(e.u, r) && !HR.get(e.v, r))
                uncMass += exp2(logq(HL, e.u, HR, e.v, r, lb));
        double rho = uncMass / Q;
        if (rho <= 0) {
            if (S.certIter < 0) { S.certIter = r + 1; S.certType = 'R'; }
            if (S.qualIter < 0) S.qualIter = r + 1;   // rho=0 => M maximum
            break;
        }
        double beta = adaptive ? min(Bcap, max(2.0, eps * (1 - rho) / (rho * (1 - eps)))) : 2.0;
        lb.push_back(log2(beta));
        Phi += eps * log2(beta) - log2(1 + (beta - 1) * rho);
        if (S.certIter < 0 && Phi > lm) { S.certIter = r + 1; S.certType = 'P'; }
        if (S.qualIter > 0 && S.certIter > 0) break;
    }
    S.ratio = (double)best / mu;
    return S;
}

// ---------------- instances ----------------
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
vector<Edge> genChungLu(int nL, int nR, size_t mTarget, double beta, mt19937_64& rng) {
    vector<double> wL(nL), wR(nR);
    for (int i = 0; i < nL; ++i) wL[i] = pow(i + 1, -beta);
    for (int i = 0; i < nR; ++i) wR[i] = pow(i + 1, -beta);
    discrete_distribution<int> DR(wR.begin(), wR.end());
    vector<double> cum(nL); partial_sum(wL.begin(), wL.end(), cum.begin());
    uniform_real_distribution<double> U(0, cum.back());
    set<pair<int,int>> seen; vector<Edge> es;
    while (es.size() < mTarget) {
        int u = lower_bound(cum.begin(), cum.end(), U(rng)) - cum.begin();
        int v = DR(rng);
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
bool loadMTX(const string& path, vector<Edge>& es, int& nL, int& nR) {
    ifstream f(path); if (!f) return false;
    string line; getline(f, line);
    if (line.find("MatrixMarket") == string::npos) return false;
    bool symmetric = line.find("symmetric") != string::npos;
    while (getline(f, line) && !line.empty() && line[0] == '%') {}
    size_t nnz; { istringstream ss(line); ss >> nL >> nR >> nnz; }
    es.clear(); es.reserve(symmetric ? 2 * nnz : nnz);
    int i, j;
    while (getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        istringstream ss(line); ss >> i >> j; --i; --j;
        if (i == j) continue;
        es.push_back({i, j});
        if (symmetric) es.push_back({j, i});
    }
    mt19937_64 g(12345); shuffle(es.begin(), es.end(), g);
    return true;
}
int exactMu(const vector<Edge>& es, int nL, int nR) {
    HopcroftKarp hk(nL, nR);
    for (const Edge& e : es) hk.addEdge(e.u, e.v);
    return hk.maxMatching();
}

struct Agg { vector<double> v; void add(double x){v.push_back(x);}
    double mean() const { return accumulate(v.begin(),v.end(),0.0)/max<size_t>(1,v.size()); }
    double sd() const { if (v.size()<2) return 0; double mu=mean(),s=0;
        for(double x:v) s+=(x-mu)*(x-mu); return sqrt(s/(v.size()-1)); } };

int main(int argc, char** argv) {
    int SEEDS = 10, MAXIT = 500;
    vector<string> mtxFiles;
    vector<double> EPS = {0.1, 0.05};             // override with --eps a,b,c
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--seeds" && i + 1 < argc) SEEDS = atoi(argv[++i]);
        else if (a == "--maxit" && i + 1 < argc) MAXIT = atoi(argv[++i]);
        else if (a == "--eps" && i + 1 < argc) {
            EPS.clear(); string s = argv[++i]; size_t p = 0, q;
            while ((q = s.find(',', p)) != string::npos) { EPS.push_back(stod(s.substr(p, q - p))); p = q + 1; }
            EPS.push_back(stod(s.substr(p)));
        }
        else mtxFiles.push_back(a);
    }
    uint64_t genSeed = 424242ULL;                 // instances fixed across all runs
    mt19937_64 grng(genSeed);
    struct Case { string name; vector<Edge> es; int nL, nR; };
    vector<Case> cases;
    cases.push_back({"hiddenPM-2000", genHidden(2000, 400, 4000, grng), 2000, 2000});
    cases.push_back({"hiddenPM-4000", genHidden(4000, 800, 4000, grng), 4000, 4000});
    cases.push_back({"RMAT-12-1M",    genRMAT(12, 1000000, grng), 4096, 4096});
    cases.push_back({"ChungLu-4096",  genChungLu(4096, 4096, 800000, 0.5, grng), 4096, 4096});
    cases.push_back({"ER-2000-d400",  genER(2000, 2000, 400, grng), 2000, 2000});
    for (auto& p : mtxFiles) {
        vector<Edge> es; int nL, nR;
        if (loadMTX(p, es, nL, nR)) {
            string base = p.substr(p.find_last_of("/\\") + 1);
            cases.push_back({base, es, nL, nR});
            fprintf(stderr, "[loaded %s: %d x %d, m=%zu]\n", base.c_str(), nL, nR, es.size());
        } else fprintf(stderr, "[FAILED to load %s]\n", p.c_str());
    }

    FILE* raw = fopen("robustness_raw.csv", "w");
    fprintf(raw, "instance,m,mu,eps,rule,seed,qualIter,qualPasses,certIter,certType,peakSamp,ratio\n");
    printf("%-16s %5s | %-8s | %-14s %-14s %-16s %-9s %-8s\n",
           "instance", "eps", "rule", "qualIter", "qualPasses", "certIter(type%%)", "ratio", "fails");
    for (auto& C : cases) {
        int mu = exactMu(C.es, C.nL, C.nR);
        // regime check
        for (double eps : EPS) {
            double thr = 2.0 * (C.nL + C.nR) / eps;
            if (C.es.size() < 1.5 * thr)
                fprintf(stderr, "[warn %s eps=%.2f: m=%zu < 1.5*(2n/eps)=%.0f — degenerate regime]\n",
                        C.name.c_str(), eps, C.es.size(), 1.5 * thr);
            for (int ad = 0; ad <= 1; ++ad) {
                Agg qi, qp, ci, ra; int fails = 0, rhoZero = 0, nCert = 0;
                for (int s = 0; s < SEEDS; ++s) {
                    RunResult R = run(C.es, C.nL, C.nR, eps, ad,
                                      0x9e3779b97f4a7c15ULL * (s + 1) ^ hash<string>{}(C.name),
                                      mu, MAXIT);
                    fprintf(raw, "%s,%zu,%d,%.2f,%s,%d,%d,%d,%d,%c,%zu,%.6f\n",
                            C.name.c_str(), C.es.size(), mu, eps, ad ? "adaptive" : "fixed2",
                            s, R.qualIter, R.qualIter > 0 ? 1 + 3 * R.qualIter : -1,
                            R.certIter, R.certType, R.peakSample, R.ratio);
                    if (R.qualIter < 0) ++fails;
                    else { qi.add(R.qualIter); qp.add(1 + 3.0 * R.qualIter); }
                    if (R.certIter > 0) { ci.add(R.certIter); ++nCert; if (R.certType == 'R') ++rhoZero; }
                    ra.add(R.ratio);
                }
                char cbuf[48];
                if (nCert) snprintf(cbuf, 48, "%.1f+-%.1f (R:%d%%)", ci.mean(), ci.sd(),
                                    (int)round(100.0 * rhoZero / nCert));
                else snprintf(cbuf, 48, "none<=maxit");
                printf("%-16s %5.2f | %-8s | %6.1f+-%-6.1f %6.1f+-%-6.1f %-16s %.4f+-%.4f %d\n",
                       C.name.c_str(), eps, ad ? "adaptive" : "fixed2",
                       qi.mean(), qi.sd(), qp.mean(), qp.sd(), cbuf, ra.mean(), ra.sd(), fails);
                fflush(stdout); fflush(raw);
            }
        }
    }
    fclose(raw);
    return 0;
}
