import os
import random
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

TL_SIZES = {}

def _safe_sh(name):
    return str(name)[:31].replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_").replace("[", "(").replace("]", ")").replace(":", "-")

def export_seed_results_to_excel(path, **sheets):
    if not path:
        return None
    path = str(path)
    if not path.lower().endswith(".xlsx"):
        path += ".xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as wr:
        for s_name, data in sheets.items():
            if data is None:
                continue
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            df.to_excel(wr, sheet_name=_safe_sh(s_name), index=False)
            ws = wr.sheets[_safe_sh(s_name)]
            for col in ws.columns:
                mx = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(mx + 2, 10), 38)
    print(f"\n[Export] Results written to: {path}")
    return path

B = 1
POP_SIZE = 100
ELITISM_RATE = 0.10
UNIFORM_MUTATION_PROB = 0.01
SWAP_MUTATION_PROB = 0.01
TOURNAMENT_RATE = 0.20
NO_IMPROVEMENT_LIMIT = 20
MAX_TIME_SECONDS = 3600.0
SETUP_TIME = 1.0
THETA_M = 72.0

def trm_solve(mag, sizes, scores, req):
    if req <= 0:
        return []
    tls = sorted(list(mag))
    n = len(tls)
    best_k = None
    best_sub = []
    for mask in range(1, 1 << n):
        sub = [tls[i] for i in range(n) if mask & (1 << i)]
        freed = sum(sizes[t] for t in sub)
        if freed < req:
            continue
        obj = sum(scores.get(t, 0) for t in sub)
        k = (obj, freed, len(sub), tuple(sub))
        if best_k is None or k < best_k:
            best_k = k
            best_sub = sub
    return best_sub

class Ind:
    def __init__(self, jv, mv):
        self.jv = list(jv)
        self.mv = list(mv)
        self.fit = float('inf')
        self.tard = 0.0
        self.su = 0
    def sig(self):
        return (tuple(self.jv), tuple(self.mv))

class Dec:
    def __init__(self, ops_by_j, num_m, cap, tau=1.0):
        self.ops_by_j = ops_by_j
        self.num_m = num_m
        self.C = cap
        self.tau = tau

    def eval_ind(self, ind):
        jv, mv = ind.jv, ind.mv
        n = len(jv)
        Tm = {m: set() for m in range(1, self.num_m + 1)}
        m_seq = {m: [] for m in range(1, self.num_m + 1)}
        tmp = {}
        for g in range(n):
            jid = jv[g]
            mid = mv[g]
            occ = tmp.get(jid, 0)
            tmp[jid] = occ + 1
            op = self.ops_by_j[jid][occ]
            t = op['tool_set']
            if t not in m_seq[mid]:
                m_seq[mid].append(t)
        for mid in range(1, self.num_m + 1):
            curr = 0
            for t in m_seq[mid]:
                sz = TL_SIZES[t]
                if curr + sz <= self.C:
                    Tm[mid].add(t)
                    curr += sz
                else:
                    break

        am = {m: 0.0 for m in range(1, self.num_m + 1)}
        fin = {}
        tot_tard, tot_su = 0.0, 0
        su_pos = []
        occs = {}
        succ = {m: [] for m in range(1, self.num_m + 1)}
        tmp2 = {}
        for g in range(n):
            jid = jv[g]
            mid = mv[g]
            occ = tmp2.get(jid, 0)
            tmp2[jid] = occ + 1
            op = self.ops_by_j[jid][occ]
            succ[mid].append((op['tool_set'], op['size']))

        for g in range(n):
            jid = jv[g]
            mid = mv[g]
            occ = occs.get(jid, 0)
            occs[jid] = occ + 1
            op = self.ops_by_j[jid][occ]
            t, sz = op['tool_set'], op['size']
            r, p, d = op['r'], op['p'], op['d']
            succ[mid].pop(0)
            z = 0
            if t not in Tm[mid]:
                z = 1
                su_pos.append(g)
                curr_sz = sum(TL_SIZES[x] for x in Tm[mid])
                free = self.C - curr_sz
                if free < sz:
                    need = sz - free
                    fut = [item[0] for item in succ[mid]]
                    fut_uq = []
                    for ft in fut:
                        if ft in Tm[mid] and ft not in fut_uq:
                            fut_uq.append(ft)
                    scores = {}
                    for ft in Tm[mid]:
                        if ft in fut_uq:
                            scores[ft] = len(fut_uq) - fut_uq.index(ft)
                        else:
                            scores[ft] = 0
                    z_sc = [x for x in Tm[mid] if scores[x] == 0]
                    z_wt = sum(TL_SIZES[x] for x in z_sc)
                    if z_wt >= need:
                        for x in z_sc:
                            Tm[mid].remove(x)
                    else:
                        for x in z_sc:
                            Tm[mid].remove(x)
                        evict = trm_solve(list(Tm[mid]), TL_SIZES, scores, need - z_wt)
                        for x in evict:
                            Tm[mid].remove(x)
                    Tm[mid].add(t)
                    tot_su += 1
                else:
                    Tm[mid].add(t)
                    tot_su += 1
            prev = fin.get((jid, occ - 1), 0.0) if occ > 0 else 0.0
            start = max(r, am[mid], prev)
            end = start + p + (self.tau * z)
            am[mid] = end
            fin[(jid, occ)] = end
            tot_tard += max(0.0, end - d)
        ind.tard = tot_tard
        ind.su = tot_su
        ind.fit = tot_tard + (self.tau * tot_su)
        ind.su_pos = su_pos

class PracHeur:
    def __init__(self, ops, num_m, cap, tau=SETUP_TIME, theta=THETA_M):
        self.O = ops
        self.M = list(range(1, num_m + 1))
        self.C = cap
        self.tau = tau
        self.theta = theta
        self.Tm = {m: set() for m in self.M}
        self.am = {m: 0.0 for m in self.M}
        self.sizes = {op['tool_set']: op['size'] for op in ops if 'tool_set' in op}

    def get_sz(self, m):
        return sum(self.sizes[t] for t in self.Tm[m])

    def run(self):
        O_sorted = sorted(self.O, key=lambda x: x['d'])
        for op in O_sorted:
            t, sz = op['tool_set'], op['size']
            mT = [m for m in self.M if t in self.Tm[m]]
            if not mT:
                MC = [m for m in self.M if (self.C - self.get_sz(m)) >= sz]
                if MC:
                    m_sel = min(MC, key=lambda m: (len(self.Tm[m]), m))
                    self.Tm[m_sel].add(t)
        tot_tard, tot_su = 0, 0
        fin, occs = {}, {}
        jv, mv = [], []
        for op in O_sorted:
            jid = op['job_id']
            t, sz = op['tool_set'], op['size']
            r, p, d = op['r'], op['p'], op['d']
            occ = occs.get(jid, 0)
            occs[jid] = occ + 1
            mP = min(self.M, key=lambda m: self.am[m])
            mT_list = [m for m in self.M if t in self.Tm[m]]
            mT = mT_list[0] if mT_list else None
            def get_st(m):
                prev = fin.get((jid, occ - 1), 0.0) if occ > 0 else 0.0
                return max(r, self.am[m], prev)
            if mT is not None:
                if mT != mP and (get_st(mT) - get_st(mP)) >= self.theta:
                    m_sel, z = mP, 1
                else:
                    m_sel, z = mT, 0
            else:
                m_sel, z = mP, 1
            if z == 1:
                req = sz - (self.C - self.get_sz(m_sel))
                while req > 0 and self.Tm[m_sel]:
                    rem = random.choice(sorted(self.Tm[m_sel]))
                    self.Tm[m_sel].remove(rem)
                    req = sz - (self.C - self.get_sz(m_sel))
                self.Tm[m_sel].add(t)
                tot_su += 1
            start = get_st(m_sel)
            end = start + p + (self.tau * z)
            self.am[m_sel] = end
            fin[(jid, occ)] = end
            tot_tard += max(0.0, end - d)
            jv.append(jid)
            mv.append(m_sel)
        ind = Ind(jv, mv)
        ind.fit = tot_tard + (self.tau * tot_su)
        return ind

class Alns:
    def __init__(self, ops, num_m, cap, tau=1.0, alpha=0.20, gamma=0.80, eps=0.30, eps_min=0.05, eps_dec=0.995, r_scale=0.05, dst_f=(0.03, 0.08), temp=None, cool=0.995, min_t=1e-6, max_ins=12, max_mc=3, max_rem=8):
        self.ops = ops
        self.num_m = num_m
        self.C = cap
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.r_scale = r_scale
        self.dst_f = dst_f
        self.temp = temp
        self.cool = cool
        self.min_t = min_t
        self.max_ins = max_ins
        self.max_mc = max_mc
        self.max_rem = max_rem
        self.cache = {}
        self.hits, self.miss, self.cur_it = 0, 0, 0
        self.ops_by_j = {}
        for op in ops:
            jid = int(op["job_id"])
            self.ops_by_j.setdefault(jid, []).append(op)
        for jid in self.ops_by_j:
            self.ops_by_j[jid].sort(key=lambda x: x["op_id"])
        self.tot_ops = {j: len(l) for j, l in self.ops_by_j.items()}
        self.dec = Dec(self.ops_by_j, self.num_m, self.C, self.tau)
        self.dst_ops = {"rand": self.dst_rand, "edd": self.dst_edd, "load": self.dst_load, "su": self.dst_su}
        self.rep_ops = {"greedy": self.rep_greedy, "regret": self.rep_regret, "edd": self.rep_edd, "load": self.rep_load}
        self.actions = [(d, r) for d in self.dst_ops for r in self.rep_ops]
        self.q_table = {}
        self.dst_w = {n: 1.0 for n in self.dst_ops}
        self.rep_w = {n: 1.0 for n in self.rep_ops}

    def clone(self, src):
        new = Ind(src.jv, src.mv)
        new.fit = src.fit
        new.tard = src.tard
        new.su = src.su
        for attr in ["rt", "gens", "stop", "hist", "su_pos", "alns_it", "alns_rt", "alns_stop", "alns_hist", "dst_w", "rep_w", "hits", "miss"]:
            if hasattr(src, attr):
                setattr(new, attr, getattr(src, attr))
        return new

    def _eval(self, jv, mv):
        k = (tuple(jv), tuple(mv))
        if k in self.cache:
            self.hits += 1
            return self.clone(self.cache[k])
        ind = Ind(jv, mv)
        self.dec.eval_ind(ind)
        self.miss += 1
        self.cache[k] = self.clone(ind)
        return ind

    def _get_state(self, cur, best, no_imp):
        ref = max(1.0, abs(best.fit))
        gap = max(0.0, (cur.fit - best.fit) / ref)
        g_bucket = "near_best" if gap <= 0.01 else "large_gap"
        t_ref = max(self.min_t, self.init_temp or max(1.0, abs(cur.fit)))
        t_ratio = max(0.0, self.temp / t_ref)
        t_bucket = "hot" if t_ratio >= 0.15 else "cold"
        s_bucket = "fresh" if no_imp < 15 else ("mild" if no_imp < 50 else "stagnated")
        return (g_bucket, t_bucket, s_bucket)

    def _ensure_state(self, s):
        if s not in self.q_table:
            self.q_table[s] = {a: 0.0 for a in self.actions}

    def _sample(self, s):
        self._ensure_state(s)
        if random.random() < self.eps:
            return random.choice(self.actions)
        q_vals = self.q_table[s]
        max_q = max(q_vals.values())
        best_a = [a for a, q in q_vals.items() if q == max_q]
        return random.choice(best_a)

    def _up_q(self, s, a, rwd, s_nxt):
        self._ensure_state(s)
        self._ensure_state(s_nxt)
        old_q = self.q_table[s][a]
        max_nxt = max(self.q_table[s_nxt].values())
        target = rwd + self.gamma * max_nxt
        self.q_table[s][a] = old_q + self.alpha * (target - old_q)

    def _decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_dec)

    def _refresh_w(self):
        d_sc = {n: [] for n in self.dst_ops}
        r_sc = {n: [] for n in self.rep_ops}
        for vals in self.q_table.values():
            for (d, r), q in vals.items():
                d_sc[d].append(q)
                r_sc[r].append(q)
        for d in self.dst_ops:
            self.dst_w[d] = max(0.05, 1.0 + (float(np.mean(d_sc[d])) if d_sc[d] else 0.0))
        for r in self.rep_ops:
            self.rep_w[r] = max(0.05, 1.0 + (float(np.mean(r_sc[r])) if r_sc[r] else 0.0))

    def _accept(self, cand, cur):
        if cand.fit <= cur.fit:
            return True
        t = max(self.min_t, self.temp)
        p = np.exp(-(cand.fit - cur.fit) / t)
        return random.random() < p

    def _get_q(self, n):
        lo, hi = self.dst_f
        frac = random.uniform(lo, hi)
        return max(1, min(n - 1, self.max_rem, int(round(frac * n))))

    def _rem_pos(self, ind, pos):
        pos = sorted(set(pos))
        rem_j = [ind.jv[i] for i in pos]
        p_j = [v for i, v in enumerate(ind.jv) if i not in pos]
        p_m = [v for i, v in enumerate(ind.mv) if i not in pos]
        return p_j, p_m, rem_j

    def _safe_pos(self, p_j, jid):
        last = -1
        for idx, j in enumerate(p_j):
            if j == jid:
                last = idx
        return list(range(last + 1, len(p_j) + 1))

    def _pos_cand(self, p_j, jid, deep=False):
        safe = self._safe_pos(p_j, jid)
        lim = 30 if deep else self.max_ins
        if len(safe) <= lim:
            return safe
        occ = p_j.count(jid)
        op = self.ops_by_j[jid][occ]
        due = float(op["d"])
        tmp = {}
        edd_pos = safe[-1]
        safe_set = set(safe)
        for idx, j in enumerate(p_j):
            k = tmp.get(j, 0)
            tmp[j] = k + 1
            edue = float(self.ops_by_j[j][k]["d"])
            if idx in safe_set and edue > due:
                edd_pos = idx
                break
        res = {safe[0], safe[-1], edd_pos}
        for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
            pos = edd_pos + delta
            if pos in safe_set:
                res.add(pos)
        rem = [p for p in safe if p not in res]
        left = max(0, lim - len(res))
        if rem and left > 0:
            res.update(random.sample(rem, min(left, len(rem))))
        return sorted(res)

    def _mach_cand(self, p_j, p_m, jid, deep=False):
        occ = p_j.count(jid)
        op = self.ops_by_j[jid][occ]
        t = op["tool_set"]
        if deep:
            return list(range(1, self.num_m + 1))
        loads = {m: 0.0 for m in range(1, self.num_m + 1)}
        has_t = {m: 0 for m in range(1, self.num_m + 1)}
        tmp = {}
        for j, m in zip(p_j, p_m):
            k = tmp.get(j, 0)
            tmp[j] = k + 1
            eop = self.ops_by_j[j][k]
            loads[m] += float(eop["p"])
            if eop["tool_set"] == t:
                has_t[m] += 1
        res = []
        match = [m for m, count in has_t.items() if count > 0]
        if match:
            res.append(max(match, key=lambda m: (has_t[m], -loads[m])))
        for m, _ in sorted(loads.items(), key=lambda kv: kv[1]):
            if m not in res:
                res.append(m)
            if len(res) >= self.max_mc:
                break
        return res

    def dst_rand(self, ind, q):
        pos = random.sample(range(len(ind.jv)), q)
        return self._rem_pos(ind, pos)

    def dst_edd(self, ind, q):
        occ = {}
        scored = []
        for idx, j in enumerate(ind.jv):
            k = occ.get(j, 0)
            occ[j] = k + 1
            op = self.ops_by_j[j][k]
            score = (-float(op["d"]), float(op["p"]), random.random())
            scored.append((score, idx))
        pos = [idx for _, idx in sorted(scored, reverse=True)[:q]]
        return self._rem_pos(ind, pos)

    def dst_load(self, ind, q):
        loads = {m: 0.0 for m in range(1, self.num_m + 1)}
        occ = {}
        for j, m in zip(ind.jv, ind.mv):
            k = occ.get(j, 0)
            occ[j] = k + 1
            loads[m] += float(self.ops_by_j[j][k]["p"])
        over = max(loads, key=loads.get)
        cand = [i for i, m in enumerate(ind.mv) if m == over]
        if len(cand) < q:
            extra = [i for i in range(len(ind.jv)) if i not in cand]
            cand += random.sample(extra, min(len(extra), q - len(cand)))
        pos = random.sample(cand, q)
        return self._rem_pos(ind, pos)

    def dst_su(self, ind, q):
        if not hasattr(ind, "su_pos"):
            self.dec.eval_ind(ind)
        su_pos = list(getattr(ind, "su_pos", []))
        if len(su_pos) >= q:
            pos = random.sample(su_pos, q)
            return self._rem_pos(ind, pos)
        pos = list(su_pos)
        rem = [i for i in range(len(ind.jv)) if i not in pos]
        if rem:
            pos += random.sample(rem, min(len(rem), q - len(pos)))
        return self._rem_pos(ind, pos)

    def _best_ins(self, p_j, p_m, jid, mc=None, deep=False):
        if mc is None:
            mc = self._mach_cand(p_j, p_m, jid, deep=deep)
        best = None
        for pos in self._pos_cand(p_j, jid, deep=deep):
            for mach in mc:
                tj = p_j[:pos] + [jid] + p_j[pos:]
                tm = p_m[:pos] + [mach] + p_m[pos:]
                cand = self._eval(tj, tm)
                k = (cand.fit, cand.tard, cand.su, pos, mach)
                if best is None or k < best[0]:
                    best = (k, cand)
        return best[1]

    def rep_greedy(self, p_j, p_m, rem_j):
        jobs = list(rem_j)
        random.shuffle(jobs)
        cj, cm = list(p_j), list(p_m)
        deep = (self.cur_it % 25 == 0) if self.cur_it > 0 else False
        for jid in jobs:
            best = self._best_ins(cj, cm, jid, deep=deep)
            cj, cm = best.jv, best.mv
        return self._eval(cj, cm)

    def rep_regret(self, p_j, p_m, rem_j):
        rem = list(rem_j)
        cj, cm = list(p_j), list(p_m)
        deep = (self.cur_it % 25 == 0) if self.cur_it > 0 else False
        while rem:
            best_ch = None
            for jid in rem:
                cands = []
                for pos in self._pos_cand(cj, jid, deep=deep):
                    for mach in self._mach_cand(cj, cm, jid, deep=deep):
                        tj = cj[:pos] + [jid] + cj[pos:]
                        tm = cm[:pos] + [mach] + cm[pos:]
                        cand = self._eval(tj, tm)
                        cands.append((cand.fit, cand))
                cands.sort(key=lambda x: x[0])
                b_fit = cands[0][0]
                s_fit = cands[1][0] if len(cands) > 1 else b_fit
                regret = s_fit - b_fit
                ch_k = (regret, -b_fit, random.random())
                if best_ch is None or ch_k > best_ch[0]:
                    best_ch = (ch_k, jid, cands[0][1])
            _, chosen_j, chosen_ind = best_ch
            cj, cm = chosen_ind.jv, chosen_ind.mv
            rem.remove(chosen_j)
        return self._eval(cj, cm)

    def rep_edd(self, p_j, p_m, rem_j):
        cj, cm = list(p_j), list(p_m)
        jobs = list(rem_j)
        deep = (self.cur_it % 25 == 0) if self.cur_it > 0 else False
        jobs.sort(key=lambda j: self.ops_by_j[j][cj.count(j)]["d"])
        for jid in jobs:
            occ = cj.count(jid)
            due = float(self.ops_by_j[jid][occ]["d"])
            cand_pos = self._pos_cand(cj, jid, deep=deep)
            safe_set = set(cand_pos)
            tmp = {}
            pos = cand_pos[-1]
            for idx, ej in enumerate(cj):
                k = tmp.get(ej, 0)
                tmp[ej] = k + 1
                edue = float(self.ops_by_j[ej][k]["d"])
                if idx in safe_set and edue > due:
                    pos = idx
                    break
            best = None
            for mach in self._mach_cand(cj, cm, jid, deep=deep):
                tj = cj[:pos] + [jid] + cj[pos:]
                tm = cm[:pos] + [mach] + cm[pos:]
                cand = self._eval(tj, tm)
                k = (cand.fit, cand.tard, cand.su, mach)
                if best is None or k < best[0]:
                    best = (k, cand)
            cj, cm = best[1].jv, best[1].mv
        return self._eval(cj, cm)

    def rep_load(self, p_j, p_m, rem_j):
        cj, cm = list(p_j), list(p_m)
        for jid in rem_j:
            loads = {m: 0.0 for m in range(1, self.num_m + 1)}
            tmp = {}
            for j, m in zip(cj, cm):
                k = tmp.get(j, 0)
                tmp[j] = k + 1
                loads[m] += float(self.ops_by_j[j][k]["p"])
            least = min(loads, key=loads.get)
            best = self._best_ins(cj, cm, jid, mc=[least], deep=False)
            cj, cm = best.jv, best.mv
        return self._eval(cj, cm)

    def _get_rwd(self, cand, cur, best, acc):
        if cand.fit < best.fit:
            base = 10.0
            ref = max(1.0, abs(best.fit))
            imp = best.fit - cand.fit
        elif cand.fit < cur.fit:
            base = 5.0
            ref = max(1.0, abs(cur.fit))
            imp = cur.fit - cand.fit
        elif acc:
            base = 1.0
            imp = 0.0
            ref = 1.0
        else:
            return 0.1
        bonus = self.r_scale * 100.0 * max(0.0, imp / ref)
        return base + bonus

    def run(self, init_sol, max_it=250, max_t=60.0, stagn=50, record_h=True, show_pr=True, pr_desc="ALNS", verbose=False):
        t0 = time.time()
        cur = self.clone(init_sol)
        self.dec.eval_ind(cur)
        best = self.clone(cur)
        if self.temp is None:
            self.temp = max(1.0, 0.05 * abs(cur.fit))
        self.init_temp = self.temp
        hist = []
        no_imp = 0
        stop = "it_lim"
        iterator = tqdm(range(1, max_it + 1), desc=pr_desc, leave=False, disable=not show_pr)
        for it in iterator:
            self.cur_it = it
            elap = time.time() - t0
            if elap >= max_t:
                stop = "t_lim"
                break
            if no_imp >= stagn:
                stop = "no_imp_lim"
                break
            s = self._get_state(cur, best, no_imp)
            a = self._sample(s)
            dst_name, rep_name = a
            q = self._get_q(len(cur.jv))
            p_j, p_m, rem_j = self.dst_ops[dst_name](cur, q)
            cand = self.rep_ops[rep_name](p_j, p_m, rem_j)
            acc = self._accept(cand, cur)
            rwd = self._get_rwd(cand, cur, best, acc)
            if acc:
                cur = cand
            if cand.fit < best.fit:
                best = self.clone(cand)
                no_imp = 0
            else:
                no_imp += 1
            s_next = self._get_state(cur, best, no_imp)
            self._up_q(s, a, rwd, s_next)
            self._decay_eps()
            self._refresh_w()
            self.temp = max(self.min_t, self.temp * self.cool)
            if show_pr and (it == 1 or it % 10 == 0 or cand.fit < best.fit):
                iterator.set_postfix(best=round(float(best.fit), 2), curr=round(float(cur.fit), 2), no_imp=int(no_imp), temp=round(float(self.temp), 3), eps=round(self.eps, 3))
            if verbose and (it == 1 or it % 25 == 0 or cand.fit < best.fit):
                print(f"alns_it={it:5d} best={best.fit:.4f} cur={cur.fit:.4f} cand={cand.fit:.4f} acc={acc} d={dst_name} r={rep_name} rwd={rwd:.3f} temp={self.temp:.4f} eps={self.eps:.3f}")
            if record_h:
                hist.append({
                    "it": it,
                    "rt": time.time() - t0,
                    "best": best.fit,
                    "cur": cur.fit,
                    "cand": cand.fit,
                    "dst": dst_name,
                    "rep": rep_name,
                    "q": q,
                    "temp": self.temp,
                    "no_imp": no_imp,
                    "hits": self.hits,
                    "miss": self.miss,
                    "dst_w": dict(self.dst_w),
                    "rep_w": dict(self.rep_w),
                })
        best.alns_it = len(hist) if record_h else it
        best.alns_rt = time.time() - t0
        best.alns_stop = stop
        best.alns_hist = hist
        best.dst_w = dict(self.dst_w)
        best.rep_w = dict(self.rep_w)
        best.hits = self.hits
        best.miss = self.miss
        best.q_table = {str(k): {f"{act[0]}+{act[1]}": float(v) for act, v in vals.items()} for k, vals in self.q_table.items()}
        return best

def load_actual_kmwe_instance(path):
    num_m = 2
    cap = 80
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        for _ in range(5):
            ln = f.readline().strip()
            if not ln:
                continue
            parts = ln.split(",")
            if len(parts) >= 2:
                k, v = parts[0].strip(), parts[1].strip()
                if k == "M":
                    num_m = int(v)
                elif k == "C":
                    cap = int(v)
    df = pd.read_csv(path, skiprows=5)
    cols = ["job_id", "op_id", "r", "p", "d", "tool_set", "size"]
    df.columns = cols
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    ops = df.to_dict(orient="records")
    TL_SIZES.clear()
    for op in ops:
        TL_SIZES[op["tool_set"]] = op["size"]
    return ops, num_m, cap

def resolve_kmwe_case_file(case_name):
    possible = [os.path.join(case_name, f"{case_name}.csv"), os.path.join(case_name, f"Base {case_name}.csv"), f"{case_name}.csv"]
    for path in possible:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Missing baseline file for: {case_name}")

def run_alns_only_on_file(case_file, seed=0, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, verbose=False, show_progress=True):
    random.seed(seed)
    np.random.seed(seed)
    jobs_data, m_case, c_case = load_actual_kmwe_instance(case_file)
    t0 = time.time()
    ph_engine = PracHeur(jobs_data, m_case, c_case)
    ph_solution = ph_engine.run()
    ph_runtime = time.time() - t0
    ops_by_job = {}
    for op in jobs_data:
        jid = int(op["job_id"])
        ops_by_job.setdefault(jid, []).append(op)
    for jid in ops_by_job:
        ops_by_job[jid].sort(key=lambda x: x["op_id"])
    decoder = Dec(ops_by_job, m_case, c_case, SETUP_TIME)
    decoder.eval_ind(ph_solution)
    alns_engine = Alns(jobs_data, m_case, c_case, SETUP_TIME)
    alns_solution = alns_engine.run(ph_solution, max_t=alns_time_seconds, max_it=alns_iterations, stagn=alns_no_improvement_limit, record_h=True, verbose=verbose, show_pr=show_progress, pr_desc=f"ALNS {os.path.basename(case_file).split('.')[0]} Seed={seed}")
    decoder.eval_ind(alns_solution)
    result = {
        "case_file": case_file,
        "seed": seed,
        "PH_fitness": ph_solution.fit,
        "PH_tardiness": ph_solution.tard,
        "PH_setups": ph_solution.su,
        "PH_runtime": ph_runtime,
        "ALNS_fitness": alns_solution.fit,
        "ALNS_tardiness": alns_solution.tard,
        "ALNS_setups": alns_solution.su,
        "ALNS_runtime": alns_solution.alns_rt,
        "ALNS_iterations": alns_solution.alns_it,
        "ALNS_stop": alns_solution.alns_stop,
        "ALNS_cache_hits": getattr(alns_solution, "hits", None),
        "ALNS_cache_misses": getattr(alns_solution, "miss", None),
        "Improvement_vs_PH_%": ((alns_solution.fit - ph_solution.fit) / max(1.0, ph_solution.fit)) * 100.0,
    }
    return alns_solution, result

def run_alns_table8_replications(num_runs=10, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, show_progress=True):
    print("\n[REPLICATION: TABLE 8 - Operational Scaling Framework on 6M140]")
    case_file = resolve_kmwe_case_file("6M140")
    full_jobs_data, m_val, c_val = load_actual_kmwe_instance(case_file)
    df_sorted = pd.DataFrame(full_jobs_data).sort_values(by="r").copy()
    rows = []
    table8_seed_records = []
    n_values = [15, 25, 30, 60, 90, 120, 140]
    n_iterator = tqdm(n_values, desc="Table 8 n-slices", disable=not show_progress)
    for n_slice in n_iterator:
        n_iterator.set_postfix(n=n_slice)
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        records = []
        seed_iterator = tqdm(range(num_runs), desc=f"Seeds n={n_slice}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            random.seed(seed)
            np.random.seed(seed)
            TL_SIZES.clear()
            for op in sliced_ops:
                TL_SIZES[op["tool_set"]] = op["size"]
            t0 = time.time()
            ph_engine = PracHeur(sliced_ops, m_val, c_val)
            ph_solution = ph_engine.run()
            ph_runtime = time.time() - t0
            ops_by_job = {}
            for op in sliced_ops:
                jid = int(op["job_id"])
                ops_by_job.setdefault(jid, []).append(op)
            for jid in ops_by_job:
                ops_by_job[jid].sort(key=lambda x: x["op_id"])
            decoder = Dec(ops_by_job, m_val, c_val, SETUP_TIME)
            decoder.eval_ind(ph_solution)
            alns_engine = Alns(sliced_ops, m_val, c_val, SETUP_TIME)
            alns_solution = alns_engine.run(ph_solution, max_t=alns_time_seconds, max_it=alns_iterations, stagn=alns_no_improvement_limit, record_h=True, verbose=False, show_pr=show_progress, pr_desc=f"ALNS n={n_slice} Seed={seed}")
            decoder.eval_ind(alns_solution)
            records.append({
                "n": n_slice,
                "seed": seed,
                "PH_fitness": ph_solution.fit,
                "PH_tardiness": ph_solution.tard,
                "PH_setups": ph_solution.su,
                "PH_runtime": ph_runtime,
                "ALNS_fitness": alns_solution.fit,
                "ALNS_tardiness": alns_solution.tard,
                "ALNS_setups": alns_solution.su,
                "ALNS_runtime": alns_solution.alns_rt,
                "ALNS_iterations": alns_solution.alns_it,
                "ALNS_stop": alns_solution.alns_stop,
                "ALNS_cache_hits": getattr(alns_solution, "hits", None),
                "ALNS_cache_misses": getattr(alns_solution, "miss", None),
                "Improvement_vs_PH_%": ((alns_solution.fit - ph_solution.fit) / max(1.0, ph_solution.fit)) * 100.0,
            })
        table8_seed_records.extend(records)
        ph = np.array([r["PH_fitness"] for r in records], dtype=float)
        alns = np.array([r["ALNS_fitness"] for r in records], dtype=float)
        rows.append({
            "n": n_slice,
            "PH_μ": round(float(np.mean(ph)), 2),
            "PH_σ": round(float(np.std(ph)), 2),
            "PH_C.T.(s)": round(float(np.mean([r["PH_runtime"] for r in records])), 3),
            "ALNS_μ": round(float(np.mean(alns)), 2),
            "ALNS_σ": round(float(np.std(alns)), 2),
            "ALNS_C.T.(s)": round(float(np.mean([r["ALNS_runtime"] for r in records])), 3),
            "ALNS_it_μ": round(float(np.mean([r["ALNS_iterations"] for r in records])), 1),
            "Gap_ALNS_vs_PH (%)": f"{((np.mean(alns) - np.mean(ph)) / max(1.0, np.mean(ph))) * 100.0:.2f}%",
            "ALNS_StopReasons": ",".join(sorted(set(r["ALNS_stop"] for r in records))),
        })
    summary = pd.DataFrame(rows)
    return summary, pd.DataFrame(table8_seed_records)

def run_alns_only_replications(num_runs=10, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, show_progress=True):
    print("\n[REPLICATION: TABLE 14 - Production Base-Case Workcenters]")
    rows = []
    table14_seed_records = []
    case_names = ["2M38", "2M46", "6M140", "6M163"]
    case_iterator = tqdm(case_names, desc="Table 14 cases", disable=not show_progress)
    for case_name in case_iterator:
        case_iterator.set_postfix(case=case_name)
        case_file = resolve_kmwe_case_file(case_name)
        records = []
        seed_iterator = tqdm(range(num_runs), desc=f"Seeds {case_name}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            _, result = run_alns_only_on_file(case_file, seed=seed, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, verbose=False, show_progress=show_progress)
            result["BaseCase"] = case_name
            result["seed"] = seed
            records.append(result)
        table14_seed_records.extend(records)
        ph = np.array([r["PH_fitness"] for r in records], dtype=float)
        alns = np.array([r["ALNS_fitness"] for r in records], dtype=float)
        rows.append({
            "BaseCase": case_name,
            "PH_μ": round(float(np.mean(ph)), 2),
            "PH_σ": round(float(np.std(ph)), 2),
            "PH_C.T.(s)": round(float(np.mean([r["PH_runtime"] for r in records])), 3),
            "ALNS_μ": round(float(np.mean(alns)), 2),
            "ALNS_σ": round(float(np.std(alns)), 2),
            "ALNS_C.T.(s)": round(float(np.mean([r["ALNS_runtime"] for r in records])), 3),
            "ALNS_it_μ": round(float(np.mean([r["ALNS_iterations"] for r in records])), 1),
            "Gap_ALNS_vs_PH (%)": f"{((np.mean(alns) - np.mean(ph)) / max(1.0, np.mean(ph))) * 100.0:.2f}%",
            "ALNS_StopReasons": ",".join(sorted(set(r["ALNS_stop"] for r in records))),
        })
    summary = pd.DataFrame(rows)
    return summary, pd.DataFrame(table14_seed_records)

def run_all_seed_experiments_to_excel(num_runs=10, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, show_progress=True, output_excel="alns_ql_table8_table14_seed_results.xlsx"):
    table8_summary, table8_seed_results = run_alns_table8_replications(num_runs=num_runs, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, show_progress=show_progress)
    table14_summary, table14_seed_results = run_alns_only_replications(num_runs=num_runs, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, show_progress=show_progress)
    
    print("\n[Table 8 Summary]")
    print(table8_summary.to_string(index=False))
    print("\n[Table 14 Summary]")
    print(table14_summary.to_string(index=False))
    
    export_seed_results_to_excel(output_excel, table8_summary=table8_summary, table8_seed_results=table8_seed_results, table14_summary=table14_summary, table14_seed_results=table14_seed_results)
    return table8_summary, table8_seed_results, table14_summary, table14_seed_results

run_all_seed_experiments_to_excel(num_runs=10)