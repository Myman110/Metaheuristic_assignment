import os
import random
import time
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

# Unified global tool registry
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

# Settings
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
    def __init__(self, ops, num_m, cap, tau=1.0, rx=0.20, dst_f=(0.03, 0.08), temp=None, cool=0.995, min_t=1e-6, max_ins=12, max_mc=3, max_rem=8):
        self.ops = ops
        self.num_m = num_m
        self.C = cap
        self.tau = tau
        self.rx = rx
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

    def _sample(self, w):
        nms = list(w.keys())
        v = np.array([max(1e-12, w[n]) for n in nms], dtype=float)
        p = v / v.sum()
        return str(np.random.choice(nms, p=p))

    def _up_w(self, w, name, rwd):
        rho = self.rx
        w[name] = (1.0 - rho) * w[name] + rho * rwd
        w[name] = max(0.05, w[name])

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
        bonus = 0.05 * 100.0 * max(0.0, imp / ref)
        return base + bonus

    def run(self, init_sol, max_it=250, max_t=60.0, stagn=50, record_h=True, show_pr=False, pr_desc="ALNS", verbose=False):
        t0 = time.time()
        cur = self.clone(init_sol)
        self.dec.eval_ind(cur)
        best = self.clone(cur)
        if self.temp is None:
            self.temp = max(1.0, 0.05 * abs(cur.fit))
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
            dst_name = self._sample(self.dst_w)
            rep_name = self._sample(self.rep_w)
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
            self._up_w(self.dst_w, dst_name, rwd)
            self._up_w(self.rep_w, rep_name, rwd)
            self.temp = max(self.min_t, self.temp * self.cool)
            if show_pr and (it == 1 or it % 10 == 0 or cand.fit < best.fit):
                iterator.set_postfix(best=round(float(best.fit), 2), curr=round(float(cur.fit), 2), no_imp=int(no_imp), temp=round(float(self.temp), 3))
            if verbose and (it == 1 or it % 25 == 0 or cand.fit < best.fit):
                print(f"alns_it={it:5d} best={best.fit:.4f} cur={cur.fit:.4f} cand={cand.fit:.4f} acc={acc} d={dst_name} r={rep_name} rwd={rwd:.3f} temp={self.temp:.4f}")
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

def run_alns_only_on_file(case_file, seed=0, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, verbose=False, show_progress=False):
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
    alns_solution = alns_engine.run(ph_solution, max_t=alns_time_seconds, max_it=alns_iterations, stagn=alns_no_improvement_limit, record_h=True, verbose=verbose, show_pr=show_progress, pr_desc=f"ALNS {seed}")
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
    print(pd.DataFrame([result]).to_string(index=False))
    return alns_solution, result

def run_alns_table8_replications(num_runs=10, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500):
    print(f" ALNS-ONLY TABLE 8 ({num_runs} RUNS) ")
    case_file = resolve_kmwe_case_file("6M140")
    full_jobs_data, m_val, c_val = load_actual_kmwe_instance(case_file)
    df_sorted = pd.DataFrame(full_jobs_data).sort_values(by="r").copy()
    rows = []
    for n_slice in [15, 25, 30, 60, 90, 120, 140]:
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        records = []
        for seed in range(num_runs):
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
            alns_solution = alns_engine.run(ph_solution, max_t=alns_time_seconds, max_it=alns_iterations, stagn=alns_no_improvement_limit, record_h=True, verbose=False)
            decoder.eval_ind(alns_solution)
            records.append({
                "PH_fitness": ph_solution.fit,
                "PH_runtime": ph_runtime,
                "ALNS_fitness": alns_solution.fit,
                "ALNS_runtime": alns_solution.alns_rt,
                "ALNS_iterations": alns_solution.alns_it,
                "ALNS_stop": alns_solution.alns_stop,
            })
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
    print("\nSummary:")
    print(summary.to_string(index=False))
    return summary

def run_alns_only_replications(num_runs=10, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500):
    print(f" ALNS SAMPLES ({num_runs} RUNS) ")
    rows = []
    for case_name in ["2M38", "2M46", "6M140", "6M163"]:
        case_file = resolve_kmwe_case_file(case_name)
        records = []
        for seed in range(num_runs):
            _, result = run_alns_only_on_file(case_file, seed=seed, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, verbose=False)
            records.append(result)
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
    print("\nSummary:")
    print(summary.to_string(index=False))
    return summary

def apmx_trans(p1, p2):
    p1_occ = {}
    mapping = {}
    for i, job in enumerate(p1):
        occ = p1_occ.get(job, 0) + 1
        p1_occ[job] = occ
        mapping[(job, occ)] = i + 1
    p2_occ = {}
    tp2 = []
    for job in p2:
        occ = p2_occ.get(job, 0) + 1
        p2_occ[job] = occ
        tp2.append(mapping[(job, occ)])
    return list(range(1, len(p1) + 1)), tp2

def pmx_cross(p1, p2):
    n = len(p1)
    cx1 = random.randint(0, n - 2)
    cx2 = random.randint(cx1 + 1, n - 1)
    child1, child2 = [None] * n, [None] * n
    child1[cx1:cx2 + 1] = p2[cx1:cx2 + 1]
    child2[cx1:cx2 + 1] = p1[cx1:cx2 + 1]
    map1 = {p2[i]: p1[i] for i in range(cx1, cx2 + 1)}
    map2 = {p1[i]: p2[i] for i in range(cx1, cx2 + 1)}
    for i in range(n):
        if i < cx1 or i > cx2:
            val1 = p1[i]
            while val1 in map1:
                val1 = map1[val1]
            child1[i] = val1
            val2 = p2[i]
            while val2 in map2:
                val2 = map2[val2]
            child2[i] = val2
    return child1, child2, cx1, cx2

def pox_edd_apply(job_vector, cx1, cx2, ops_by_job):
    n = len(job_vector)
    occ_counts = {}
    outside_elements = []
    for idx, job in enumerate(job_vector):
        occ = occ_counts.get(job, 0)
        occ_counts[job] = occ + 1
        if idx < cx1 or idx > cx2:
            outside_elements.append((job, ops_by_job[job][occ], idx))
    outside_elements.sort(key=lambda x: x[1]["d"])
    new_job_vector = list(job_vector)
    outside_indices = [i for i in range(n) if i < cx1 or i > cx2]
    for idx, (job, _, _) in zip(outside_indices, outside_elements):
        new_job_vector[idx] = job
    return new_job_vector

def pox_mach_build(job_vector, ops_by_job, num_machines, magazine_capacity):
    Tm = {m: set() for m in range(1, num_machines + 1)}
    p_m = {m: 0.0 for m in range(1, num_machines + 1)}
    mach_vec = []
    occ_counts = {}
    for job in job_vector:
        occ = occ_counts.get(job, 0)
        occ_counts[job] = occ + 1
        op_data = ops_by_job[job][occ]
        t_ij, phi_t, p_ij = op_data["tool_set"], op_data["size"], op_data["p"]
        m_T_list = [m for m in range(1, num_machines + 1) if t_ij in Tm[m]]
        if m_T_list:
            m_star = m_T_list[0]
        else:
            MC = []
            for m in range(1, num_machines + 1):
                current_size = sum(TL_SIZES[t] for t in Tm[m])
                if magazine_capacity - current_size >= phi_t:
                    MC.append(m)
            m_star = min(MC, key=lambda m: p_m[m]) if MC else min(range(1, num_machines + 1), key=lambda m: p_m[m])
        p_m[m_star] += p_ij
        mach_vec.append(m_star)
    return mach_vec

class MathGA:
    def __init__(self, jobs_data, num_m, cap, setup_time=SETUP_TIME):
        self.num_m = num_m
        self.C = cap
        self.tau = setup_time
        self.jobs_data = jobs_data
        self.ops_by_j = {}
        self.flat_ops = []
        for op in jobs_data:
            job_id = int(op["job_id"])
            self.ops_by_j.setdefault(job_id, []).append(op)
            self.flat_ops.append(job_id)
        for job_id in self.ops_by_j:
            self.ops_by_j[job_id].sort(key=lambda x: x["op_id"])
        self.decoder = Dec(self.ops_by_j, self.num_m, self.C, self.tau)

    def run(self, max_time_seconds=MAX_TIME_SECONDS, no_improvement_limit=NO_IMPROVEMENT_LIMIT, pop_size=POP_SIZE, tournament_rate=TOURNAMENT_RATE, elitism_rate=ELITISM_RATE, swap_mutation_prob=SWAP_MUTATION_PROB, uniform_mutation_prob=UNIFORM_MUTATION_PROB, pox_iterations=B, max_generations=None, record_history=True, verbose=False):
        population = []
        ph_engine = PracHeur(self.jobs_data, self.num_m, self.C, self.tau)
        ph_baseline = ph_engine.run()
        self.decoder.eval_ind(ph_baseline)
        population.append(ph_baseline)
        n_ops = len(self.flat_ops)
        for _ in range(pop_size - 1):
            rand_job_vec = list(self.flat_ops)
            random.shuffle(rand_job_vec)
            rand_mach_vec = [random.randint(1, self.num_m) for _ in range(n_ops)]
            ind = Ind(rand_job_vec, rand_mach_vec)
            self.decoder.eval_ind(ind)
            population.append(ind)
        best_ind = min(population, key=lambda x: x.fit)
        f_best = best_ind.fit
        no_improve = 0
        q = pox_iterations + 1
        best_improved = False
        generation = 0
        start_clock = time.time()
        history = []
        stop_reason = None
        if record_history:
            history.append({"generation": 0, "runtime": 0.0, "best_fitness": float(f_best), "current_best": float(f_best), "no_improve": 0, "used_pox": False, "improved": True})
        while True:
            elapsed = time.time() - start_clock
            if elapsed >= max_time_seconds:
                stop_reason = "time_limit"
                break
            if no_improve >= no_improvement_limit:
                stop_reason = "no_improvement_limit"
                break
            if max_generations is not None and generation >= max_generations:
                stop_reason = "debug_generation_limit"
                break
            offspring = []
            use_pox = best_improved or q <= pox_iterations
            while len(offspring) < pop_size:
                p1, p2 = self.select_parents(population, tournament_rate)
                if use_pox:
                    tp1, tp2 = apmx_trans(p1.jv, p2.jv)
                    tc1, tc2, cx1, cx2 = pmx_cross(tp1, tp2)
                    c1_job = [p1.jv[v - 1] for v in tc1]
                    c2_job = [p1.jv[v - 1] for v in tc2]
                    c1_job = pox_edd_apply(c1_job, cx1, cx2, self.ops_by_j)
                    c2_job = pox_edd_apply(c2_job, cx1, cx2, self.ops_by_j)
                    c1_mach = pox_mach_build(c1_job, self.ops_by_j, self.num_m, self.C)
                    c2_mach = pox_mach_build(c2_job, self.ops_by_j, self.num_m, self.C)
                else:
                    tp1, tp2 = apmx_trans(p1.jv, p2.jv)
                    tc1, tc2, _, _ = pmx_cross(tp1, tp2)
                    c1_job = [p1.jv[v - 1] for v in tc1]
                    c2_job = [p1.jv[v - 1] for v in tc2]
                    c1_mach, c2_mach = self.two_point_crossover(p1.mv, p2.mv)
                child1, child2 = Ind(c1_job, c1_mach), Ind(c2_job, c2_mach)
                self.apply_mutation(child1, swap_mutation_prob, uniform_mutation_prob)
                self.apply_mutation(child2, swap_mutation_prob, uniform_mutation_prob)
                self.decoder.eval_ind(child1)
                self.decoder.eval_ind(child2)
                offspring.extend([child1, child2])
            offspring = offspring[:pop_size]
            se = int(elitism_rate * pop_size)
            parents_elite = sorted(population, key=lambda x: x.fit)[:se]
            next_pop = offspring[:]
            if se > 0:
                replace_idx = random.sample(range(pop_size), se)
                for idx, elite in zip(replace_idx, parents_elite):
                    next_pop[idx] = elite
            unique_signatures = set()
            final_pop = []
            for ind in next_pop:
                sig = ind.sig()
                if sig not in unique_signatures:
                    unique_signatures.add(sig)
                    final_pop.append(ind)
                else:
                    rand_job_vec = list(self.flat_ops)
                    random.shuffle(rand_job_vec)
                    rand_mach_vec = [random.randint(1, self.num_m) for _ in range(n_ops)]
                    immigrant = Ind(rand_job_vec, rand_mach_vec)
                    self.decoder.eval_ind(immigrant)
                    final_pop.append(immigrant)
            population = final_pop
            current_best = min(population, key=lambda x: x.fit)
            improved = current_best.fit < f_best
            if improved:
                f_best = current_best.fit
                best_ind = current_best
                best_improved = True
                q = 1
                no_improve = 0
            else:
                best_improved = False
                q += 1
                no_improve += 1
            generation += 1
            elapsed = time.time() - start_clock
            if record_history:
                history.append({"generation": generation, "runtime": float(elapsed), "best_fitness": float(f_best), "current_best": float(current_best.fit), "no_improve": int(no_improve), "used_pox": bool(use_pox), "improved": bool(improved)})
            if verbose:
                print(f"gen={generation:4d} best={f_best:.4f} current={current_best.fit:.4f} no_improve={no_improve:2d} pox={use_pox} time={elapsed:.2f}s")
        if stop_reason is None:
            stop_reason = "unknown"
        best_ind.generations = generation
        best_ind.runtime = time.time() - start_clock
        best_ind.stop_reason = stop_reason
        best_ind.history = history
        return best_ind

    def select_parents(self, population, tournament_rate):
        size = max(2, int(tournament_rate * len(population)))
        return min(random.sample(population, size), key=lambda x: x.fit), min(random.sample(population, size), key=lambda x: x.fit)

    def two_point_crossover(self, m1, m2):
        n = len(m1)
        cx1 = random.randint(0, n - 2)
        cx2 = random.randint(cx1 + 1, n - 1)
        c1, c2 = list(m1), list(m2)
        c1[cx1:cx2 + 1], c2[cx1:cx2 + 1] = m2[cx1:cx2 + 1], m1[cx1:cx2 + 1]
        return c1, c2

    def apply_mutation(self, individual, Ps, Pu):
        n = len(individual.jv)
        for idx in range(n):
            if random.random() < Ps:
                j = random.randrange(n)
                individual.jv[idx], individual.jv[j] = individual.jv[j], individual.jv[idx]
        for idx in range(len(individual.mv)):
            if random.random() < Pu:
                individual.mv[idx] = random.randint(1, self.num_m)

def clone_individual(ind):
    new = Ind(ind.jv, ind.mv)
    new.fit = float(getattr(ind, "fit", float("inf")))
    new.tard = float(getattr(ind, "tard", 0.0))
    new.su = int(getattr(ind, "su", 0))
    for attr in ["runtime", "generations", "stop_reason", "history", "su_pos", "alns_iterations", "alns_runtime", "alns_stop_reason", "alns_history", "alns_destroy_weights", "alns_repair_weights", "alns_eval_cache_hits", "alns_eval_cache_misses", "source_label", "source_rank", "source_diversity"]:
        if hasattr(ind, attr):
            setattr(new, attr, getattr(ind, attr))
    return new

def unique_sorted_individuals(individuals):
    unique = {}
    for ind in individuals:
        sig = ind.sig()
        if sig not in unique or ind.fit < unique[sig].fit:
            unique[sig] = clone_individual(ind)
    return sorted(unique.values(), key=lambda x: (x.fit, x.tard, x.su))

def select_best_ga_solution(archive):
    ranked = unique_sorted_individuals(archive)
    if not ranked:
        raise ValueError("Cannot select GA solution: archive is empty.")
    best = clone_individual(ranked[0])
    best.source_rank = 1
    best.source_label = "GA_best"
    best.source_diversity = 0.0
    return [best]

class MatheuristicWithArchive(MathGA):
    def run_with_archive(self, max_time_seconds=MAX_TIME_SECONDS, no_improvement_limit=NO_IMPROVEMENT_LIMIT, pop_size=POP_SIZE, tournament_rate=TOURNAMENT_RATE, elitism_rate=ELITISM_RATE, swap_mutation_prob=SWAP_MUTATION_PROB, uniform_mutation_prob=UNIFORM_MUTATION_PROB, pox_iterations=B, max_generations=None, record_history=True, verbose=False, archive_top_per_generation=25, show_progress=False, progress_desc="GA/MH"):
        population = []
        archive = []
        ph_engine = PracHeur(self.jobs_data, self.num_m, self.C, self.tau)
        ph_baseline = ph_engine.run()
        self.decoder.eval_ind(ph_baseline)
        population.append(ph_baseline)
        archive.append(clone_individual(ph_baseline))
        n_ops = len(self.flat_ops)
        for _ in range(pop_size - 1):
            rand_job_vec = list(self.flat_ops)
            random.shuffle(rand_job_vec)
            rand_mach_vec = [random.randint(1, self.num_m) for _ in range(n_ops)]
            ind = Ind(rand_job_vec, rand_mach_vec)
            self.decoder.eval_ind(ind)
            population.append(ind)
        archive.extend(clone_individual(ind) for ind in sorted(population, key=lambda x: x.fit)[:archive_top_per_generation])
        best_ind = min(population, key=lambda x: x.fit)
        f_best = best_ind.fit
        no_improve = 0
        q = pox_iterations + 1
        best_improved = False
        generation = 0
        start_clock = time.time()
        history = []
        stop_reason = None
        pbar = tqdm(total=max_generations if max_generations is not None else None, desc=progress_desc, leave=False, disable=not show_progress)
        if record_history:
            history.append({"generation": 0, "runtime": 0.0, "best_fitness": float(f_best), "current_best": float(f_best), "no_improve": 0, "used_pox": False, "improved": True, "archive_size": len(unique_sorted_individuals(archive))})
        while True:
            elapsed = time.time() - start_clock
            if elapsed >= max_time_seconds:
                stop_reason = "time_limit"
                break
            if no_improve >= no_improvement_limit:
                stop_reason = "no_improvement_limit"
                break
            if max_generations is not None and generation >= max_generations:
                stop_reason = "debug_generation_limit"
                break
            offspring = []
            use_pox = best_improved or q <= pox_iterations
            while len(offspring) < pop_size:
                p1, p2 = self.select_parents(population, tournament_rate)
                if use_pox:
                    tp1, tp2 = apmx_trans(p1.jv, p2.jv)
                    tc1, tc2, cx1, cx2 = pmx_cross(tp1, tp2)
                    c1_job = [p1.jv[v - 1] for v in tc1]
                    c2_job = [p1.jv[v - 1] for v in tc2]
                    c1_job = pox_edd_apply(c1_job, cx1, cx2, self.ops_by_j)
                    c2_job = pox_edd_apply(c2_job, cx1, cx2, self.ops_by_j)
                    c1_mach = pox_mach_build(c1_job, self.ops_by_j, self.num_m, self.C)
                    c2_mach = pox_mach_build(c2_job, self.ops_by_j, self.num_m, self.C)
                else:
                    tp1, tp2 = apmx_trans(p1.jv, p2.jv)
                    tc1, tc2, _, _ = pmx_cross(tp1, tp2)
                    c1_job = [p1.jv[v - 1] for v in tc1]
                    c2_job = [p1.jv[v - 1] for v in tc2]
                    c1_mach, c2_mach = self.two_point_crossover(p1.mv, p2.mv)
                child1, child2 = Ind(c1_job, c1_mach), Ind(c2_job, c2_mach)
                self.apply_mutation(child1, swap_mutation_prob, uniform_mutation_prob)
                self.apply_mutation(child2, swap_mutation_prob, uniform_mutation_prob)
                self.decoder.eval_ind(child1)
                self.decoder.eval_ind(child2)
                offspring.extend([child1, child2])
            offspring = offspring[:pop_size]
            se = int(elitism_rate * pop_size)
            parents_elite = sorted(population, key=lambda x: x.fit)[:se]
            next_pop = offspring[:]
            if se > 0:
                replace_idx = random.sample(range(pop_size), se)
                for idx, elite in zip(replace_idx, parents_elite):
                    next_pop[idx] = elite
            unique_signatures = set()
            final_pop = []
            for ind in next_pop:
                sig = ind.sig()
                if sig not in unique_signatures:
                    unique_signatures.add(sig)
                    final_pop.append(ind)
                else:
                    rand_job_vec = list(self.flat_ops)
                    random.shuffle(rand_job_vec)
                    rand_mach_vec = [random.randint(1, self.num_m) for _ in range(n_ops)]
                    immigrant = Ind(rand_job_vec, rand_mach_vec)
                    self.decoder.eval_ind(immigrant)
                    final_pop.append(immigrant)
            population = final_pop
            archive.extend(clone_individual(ind) for ind in sorted(population, key=lambda x: x.fit)[:archive_top_per_generation])
            current_best = min(population, key=lambda x: x.fit)
            improved = current_best.fit < f_best
            if improved:
                f_best = current_best.fit
                best_ind = current_best
                best_improved = True
                q = 1
                no_improve = 0
            else:
                best_improved = False
                q += 1
                no_improve += 1
            generation += 1
            elapsed = time.time() - start_clock
            if show_progress:
                pbar.update(1)
                pbar.set_postfix(best=round(float(f_best), 2), noimp=int(no_improve), archive=len(unique_sorted_individuals(archive)))
            if record_history:
                history.append({"generation": generation, "runtime": float(elapsed), "best_fitness": float(f_best), "current_best": float(current_best.fit), "no_improve": int(no_improve), "used_pox": bool(use_pox), "improved": bool(improved), "archive_size": len(unique_sorted_individuals(archive))})
            if verbose:
                print(f"gen={generation:4d} best={f_best:.4f} current={current_best.fit:.4f} no_improve={no_improve:2d} pox={use_pox} time={elapsed:.2f}s archive={len(unique_sorted_individuals(archive))}")
        pbar.close()
        if stop_reason is None:
            stop_reason = "unknown"
        best_ind = clone_individual(best_ind)
        best_ind.generations = generation
        best_ind.runtime = time.time() - start_clock
        best_ind.stop_reason = stop_reason
        best_ind.history = history
        best_ind.ga_archive = unique_sorted_individuals(archive)
        return best_ind, best_ind.ga_archive

def build_ops_by_job(jobs_data):
    ops_by_job = {}
    for op in jobs_data:
        job_id = int(op["job_id"])
        ops_by_job.setdefault(job_id, []).append(op)
    for job_id in ops_by_job:
        ops_by_job[job_id].sort(key=lambda x: x["op_id"])
    return ops_by_job

def run_ph_mh_alns_aos_on_file(case_file, seed=0, mh_time_seconds=MAX_TIME_SECONDS, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, mh_pop_size=POP_SIZE, mh_max_generations=None, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, reset_seed_per_alns_start=True, verbose=False, show_progress=True):
    random.seed(seed)
    np.random.seed(seed)
    jobs_data, m_case, c_case = load_actual_kmwe_instance(case_file)
    ops_by_job = build_ops_by_job(jobs_data)
    decoder = Dec(ops_by_job, m_case, c_case, SETUP_TIME)
    t0 = time.time()
    ph_engine = PracHeur(jobs_data, m_case, c_case)
    ph_solution = ph_engine.run()
    decoder.eval_ind(ph_solution)
    ph_runtime = time.time() - t0
    mh_engine = MatheuristicWithArchive(jobs_data, m_case, c_case)
    mh_solution, ga_archive = mh_engine.run_with_archive(max_time_seconds=mh_time_seconds, no_improvement_limit=mh_no_improvement_limit, pop_size=mh_pop_size, max_generations=mh_max_generations, record_history=True, verbose=verbose, show_progress=show_progress, progress_desc=f"ALNS-only seed={seed}")
    decoder.eval_ind(mh_solution)
    starts = select_best_ga_solution(ga_archive)
    alns_results = []
    start_iterator = tqdm(list(enumerate(starts, start=1)), desc=f"ALNS starts seed={seed}", leave=False, disable=not show_progress)
    for idx, start_ind in start_iterator:
        if reset_seed_per_alns_start:
            alns_seed = seed * 1000 + idx
            random.seed(alns_seed)
            np.random.seed(alns_seed)
        start_ind = clone_individual(start_ind)
        decoder.eval_ind(start_ind)
        if show_progress:
            start_iterator.set_postfix(start=getattr(start_ind, "source_label", f"GA_start_{idx}"), fit=round(float(start_ind.fit), 2))
        alns_engine = Alns(jobs_data, m_case, c_case, SETUP_TIME)
        alns_solution = alns_engine.run(start_ind, max_it=alns_iterations, max_t=alns_time_seconds, stagn=alns_no_improvement_limit, record_h=True, show_pr=show_progress, pr_desc=f"ALNS seed={seed} start={idx}/{len(starts)}")
        decoder.eval_ind(alns_solution)
        alns_solution.start_label = getattr(start_ind, "source_label", f"GA_start_{idx}")
        alns_solution.start_rank = idx
        alns_solution.start_fitness = float(start_ind.fit)
        alns_solution.start_tardiness = float(start_ind.tard)
        alns_solution.start_setups = int(start_ind.su)
        alns_solution.start_diversity = float(getattr(start_ind, "source_diversity", 0.0))
        alns_results.append(alns_solution)
    best_hybrid = min(alns_results, key=lambda x: (x.fit, x.tard, x.su))
    rows = []
    for sol in alns_results:
        rows.append({
            "case_file": case_file,
            "seed": seed,
            "start_rank": sol.start_rank,
            "start_label": sol.start_label,
            "start_fitness": sol.start_fitness,
            "start_tardiness": sol.start_tardiness,
            "start_setups": sol.start_setups,
            "start_diversity": sol.start_diversity,
            "ALNS_fitness": sol.fit,
            "ALNS_tardiness": sol.tard,
            "ALNS_setups": sol.su,
            "ALNS_runtime": getattr(sol, "alns_rt", np.nan),
            "ALNS_iterations": getattr(sol, "alns_it", np.nan),
            "ALNS_stop_reason": getattr(sol, "alns_stop", "unknown"),
            "is_best_hybrid_start": sol is best_hybrid,
        })
    summary = {
        "case_file": case_file,
        "seed": seed,
        "PH_fitness": ph_solution.fit,
        "PH_tardiness": ph_solution.tard,
        "PH_setups": ph_solution.su,
        "PH_runtime": ph_runtime,
        "MH_fitness": mh_solution.fit,
        "MH_tardiness": mh_solution.tard,
        "MH_setups": mh_solution.su,
        "MH_runtime": getattr(mh_solution, "runtime", np.nan),
        "MH_generations": getattr(mh_solution, "generations", np.nan),
        "MH_stop_reason": getattr(mh_solution, "stop_reason", "unknown"),
        "GA_archive_unique_size": len(ga_archive),
        "Hybrid_fitness": best_hybrid.fit,
        "Hybrid_tardiness": best_hybrid.tard,
        "Hybrid_setups": best_hybrid.su,
        "ILP_final_fitness": best_hybrid.fit,
        "ILP_final_tardiness": best_hybrid.tard,
        "ILP_final_setups": best_hybrid.su,
        "Hybrid_ALNS_runtime": getattr(best_hybrid, "alns_rt", np.nan),
        "Hybrid_total_runtime": ph_runtime + getattr(mh_solution, "runtime", 0.0) + sum(getattr(s, "alns_rt", 0.0) for s in alns_results),
        "Best_ALNS_start_label": getattr(best_hybrid, "start_label", "unknown"),
        "Best_ALNS_start_rank": getattr(best_hybrid, "start_rank", np.nan),
    }
    return summary, rows, {
        "ph_solution": ph_solution,
        "mh_solution": mh_solution,
        "ga_starts": starts,
        "alns_results": alns_results,
        "best_hybrid": best_hybrid,
        "ga_archive": ga_archive,
    }

def run_hybrid_experiments(case_files, seeds=range(10), show_progress=True, **kwargs):
    all_summaries = []
    all_start_rows = []
    case_iterator = tqdm(list(case_files), desc="Cases", disable=not show_progress)
    for case_file in case_iterator:
        case_iterator.set_postfix(case=os.path.basename(str(case_file)))
        seed_iterator = tqdm(list(seeds), desc=f"Seeds {os.path.basename(str(case_file))}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            msg = f"Running strict PH -> GA/MH -> ALNS-AOS -> ILP | case={case_file} | seed={seed}"
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)
            summary, start_rows, _ = run_ph_mh_alns_aos_on_file(case_file=case_file, seed=int(seed), show_progress=show_progress, **kwargs)
            all_summaries.append(summary)
            all_start_rows.extend(start_rows)
    summary_df = pd.DataFrame(all_summaries)
    starts_df = pd.DataFrame(all_start_rows)
    print("\nSummary results:")
    print(summary_df.to_string(index=False))
    return summary_df, starts_df

def run_ph_mh_alns_aos_on_jobs_data(jobs_data, num_machines, magazine_capacity, case_label="custom_instance", seed=0, mh_time_seconds=MAX_TIME_SECONDS, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, mh_pop_size=POP_SIZE, mh_max_generations=None, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, reset_seed_per_alns_start=True, verbose=False, show_progress=True):
    random.seed(seed)
    np.random.seed(seed)
    TL_SIZES.clear()
    TL_SIZES.update({op["tool_set"]: op["size"] for op in jobs_data})
    ops_by_job = build_ops_by_job(jobs_data)
    decoder = Dec(ops_by_job, num_machines, magazine_capacity, SETUP_TIME)
    t0 = time.time()
    ph_engine = PracHeur(jobs_data, num_machines, magazine_capacity)
    ph_solution = ph_engine.run()
    decoder.eval_ind(ph_solution)
    ph_runtime = time.time() - t0
    mh_engine = MatheuristicWithArchive(jobs_data, num_machines, magazine_capacity)
    mh_solution, ga_archive = mh_engine.run_with_archive(max_time_seconds=mh_time_seconds, no_improvement_limit=mh_no_improvement_limit, pop_size=mh_pop_size, max_generations=mh_max_generations, record_history=True, verbose=verbose, show_progress=show_progress, progress_desc=f"GA/MH {case_label} seed={seed}")
    decoder.eval_ind(mh_solution)
    starts = select_best_ga_solution(ga_archive)
    alns_results = []
    start_iterator = tqdm(list(enumerate(starts, start=1)), desc=f"ALNS starts {case_label} seed={seed}", leave=False, disable=not show_progress)
    for idx, start_ind in start_iterator:
        if reset_seed_per_alns_start:
            alns_seed = seed * 1000 + idx
            random.seed(alns_seed)
            np.random.seed(alns_seed)
        start_ind = clone_individual(start_ind)
        decoder.eval_ind(start_ind)
        if show_progress:
            start_iterator.set_postfix(start=getattr(start_ind, "source_label", f"GA_start_{idx}"), fit=round(float(start_ind.fit), 2))
        alns_engine = Alns(jobs_data, num_machines, magazine_capacity, SETUP_TIME)
        alns_solution = alns_engine.run(start_ind, max_it=alns_iterations, max_t=alns_time_seconds, stagn=alns_no_improvement_limit, record_h=True, show_pr=show_progress, pr_desc=f"ALNS {case_label} seed={seed} start={idx}/{len(starts)}")
        decoder.eval_ind(alns_solution)
        alns_solution.start_label = getattr(start_ind, "source_label", f"GA_start_{idx}")
        alns_solution.start_rank = idx
        alns_solution.start_fitness = float(start_ind.fit)
        alns_solution.start_tardiness = float(start_ind.tard)
        alns_solution.start_setups = int(start_ind.su)
        alns_solution.start_diversity = float(getattr(start_ind, "source_diversity", 0.0))
        alns_results.append(alns_solution)
    best_hybrid = min(alns_results, key=lambda x: (x.fit, x.tard, x.su))
    start_rows = []
    for sol in alns_results:
        start_rows.append({
            "case_label": case_label,
            "seed": seed,
            "start_rank": sol.start_rank,
            "start_label": sol.start_label,
            "start_fitness": sol.start_fitness,
            "start_tardiness": sol.start_tardiness,
            "start_setups": sol.start_setups,
            "start_diversity": sol.start_diversity,
            "ALNS_fitness": sol.fit,
            "ALNS_tardiness": sol.tard,
            "ALNS_setups": sol.su,
            "ALNS_runtime": getattr(sol, "alns_rt", np.nan),
            "ALNS_iterations": getattr(sol, "alns_it", np.nan),
            "ALNS_stop_reason": getattr(sol, "alns_stop", "unknown"),
            "is_best_hybrid_start": sol is best_hybrid,
        })
    summary = {
        "case_label": case_label,
        "seed": seed,
        "PH_fitness": ph_solution.fit,
        "PH_tardiness": ph_solution.tard,
        "PH_setups": ph_solution.su,
        "PH_runtime": ph_runtime,
        "MH_fitness": mh_solution.fit,
        "MH_tardiness": mh_solution.tard,
        "MH_setups": mh_solution.su,
        "MH_runtime": getattr(mh_solution, "runtime", np.nan),
        "MH_generations": getattr(mh_solution, "generations", np.nan),
        "MH_stop_reason": getattr(mh_solution, "stop_reason", "unknown"),
        "GA_archive_unique_size": len(ga_archive),
        "Hybrid_fitness": best_hybrid.fit,
        "Hybrid_tardiness": best_hybrid.tard,
        "Hybrid_setups": best_hybrid.su,
        "ILP_final_fitness": best_hybrid.fit,
        "ILP_final_tardiness": best_hybrid.tard,
        "ILP_final_setups": best_hybrid.su,
        "Hybrid_ALNS_runtime": getattr(best_hybrid, "alns_rt", np.nan),
        "Hybrid_total_runtime": ph_runtime + getattr(mh_solution, "runtime", 0.0) + sum(getattr(s, "alns_rt", 0.0) for s in alns_results),
        "Best_ALNS_start_label": getattr(best_hybrid, "start_label", "unknown"),
        "Best_ALNS_start_rank": getattr(best_hybrid, "start_rank", np.nan),
    }
    return summary, start_rows, {
        "ph_solution": ph_solution,
        "mh_solution": mh_solution,
        "ga_starts": starts,
        "alns_results": alns_results,
        "best_hybrid": best_hybrid,
        "ga_archive": ga_archive,
    }

def _aggregate_hybrid_records(records, group_col):
    rows = []
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    for key, g in df.groupby(group_col, sort=False):
        ph = g["PH_fitness"].astype(float)
        mh = g["MH_fitness"].astype(float)
        hy = g["Hybrid_fitness"].astype(float)
        ph_mean = float(ph.mean())
        mh_mean = float(mh.mean())
        hy_mean = float(hy.mean())
        means = {"PH": ph_mean, "MH": mh_mean, "ALNS": hy_mean}
        best_method = min(means, key=means.get)
        rows.append({
            group_col: key,
            "PH_μ": round(ph_mean, 2),
            "PH_σ": round(float(ph.std(ddof=0)), 2),
            "PH_best": round(float(ph.min()), 2),
            "PH_C.T.(s)": round(float(g["PH_runtime"].mean()), 3),
            "MH_μ": round(mh_mean, 2),
            "MH_σ": round(float(mh.std(ddof=0)), 2),
            "MH_best": round(float(mh.min()), 2),
            "MH_C.T.(s)": round(float(g["MH_runtime"].mean()), 3),
            "MH_gen_μ": round(float(g["MH_generations"].mean()), 1),
            "ALNS_μ": round(hy_mean, 2),
            "ALNS_σ": round(float(hy.std(ddof=0)), 2),
            "ALNS_best": round(float(hy.min()), 2),
            "ALNS_C.T.(s)": round(float(g["Hybrid_total_runtime"].mean()), 3),
            "ALNS_phase_C.T.(s)": round(float(g["Hybrid_ALNS_runtime"].mean()), 3),
            "ALNS_it_μ": round(float(g["Hybrid_ALNS_iterations"].mean()), 1),
            "StopReasons": ",".join(sorted(set(map(str, g["MH_stop_reason"])))),
            "ALNS_StopReasons": ",".join(sorted(set(map(str, g["Hybrid_ALNS_stop_reason"])))),
            "Gap_MH_vs_PH (%)": f"{((mh_mean - ph_mean) / max(1.0, ph_mean)) * 100.0:.2f}%",
            "Gap_ALNS_vs_PH (%)": f"{((hy_mean - ph_mean) / max(1.0, ph_mean)) * 100.0:.2f}%",
            "Gap_ALNS_vs_MH (%)": f"{((hy_mean - mh_mean) / max(1.0, mh_mean)) * 100.0:.2f}%",
            "Best_Method": best_method,
            "Best_μ": round(float(means[best_method]), 2),
        })
    return pd.DataFrame(rows)

def _paper_like_table8_from_hybrid(run_records):
    full = _aggregate_hybrid_records(run_records, "n")
    if full.empty:
        return full
    cols = ["n", "PH_μ", "PH_σ", "PH_best", "PH_C.T.(s)", "MH_μ", "MH_σ", "MH_best", "MH_C.T.(s)", "MH_gen_μ", "StopReasons", "Gap_MH_vs_PH (%)", "ALNS_μ", "ALNS_σ", "ALNS_best", "ALNS_C.T.(s)", "ALNS_phase_C.T.(s)", "ALNS_it_μ", "ALNS_StopReasons", "Gap_ALNS_vs_PH (%)", "Gap_ALNS_vs_MH (%)", "Best_Method", "Best_μ"]
    return full[cols]

def _paper_like_table14_from_hybrid(run_records):
    full = _aggregate_hybrid_records(run_records, "BaseCase")
    if full.empty:
        return full
    cols = ["BaseCase", "PH_μ", "PH_σ", "PH_best", "PH_C.T.(s)", "MH_μ", "MH_σ", "MH_best", "MH_C.T.(s)", "MH_gen_μ", "StopReasons", "Gap_MH_vs_PH (%)", "ALNS_μ", "ALNS_σ", "ALNS_best", "ALNS_C.T.(s)", "ALNS_phase_C.T.(s)", "ALNS_it_μ", "ALNS_StopReasons", "Gap_ALNS_vs_PH (%)", "Gap_ALNS_vs_MH (%)", "Best_Method", "Best_μ"]
    return full[cols].rename(columns={"Gap_MH_vs_PH (%)": "Net_Gap_MH (%)"})

def _enrich_summary_with_best_alns_metadata(summary, objects):
    best_hybrid = objects["best_hybrid"]
    summary["Hybrid_ALNS_iterations"] = getattr(best_hybrid, "alns_it", np.nan)
    summary["Hybrid_ALNS_stop_reason"] = getattr(best_hybrid, "alns_stop", "unknown")
    return summary

def run_hybrid_table8_replications(num_runs=10, mh_time_seconds=MAX_TIME_SECONDS, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, mh_pop_size=POP_SIZE, mh_max_generations=None, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, show_progress=True, verbose=False):
    print("\n[EXACT REPLICATION: TABLE 8 - Operational Scaling Framework on 6M140]")
    case_file = resolve_kmwe_case_file("6M140")
    full_jobs_data, m_val, c_val = load_actual_kmwe_instance(case_file)
    df_sorted = pd.DataFrame(full_jobs_data).sort_values(by="r").copy()
    run_records = []
    start_records = []
    n_values = [15, 25, 30, 60, 90, 120, 140]
    n_iterator = tqdm(n_values, desc="Table 8 n-slices", disable=not show_progress)
    for n_slice in n_iterator:
        n_iterator.set_postfix(n=n_slice)
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        seed_iterator = tqdm(range(num_runs), desc=f"Seeds n={n_slice}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            random.seed(int(seed))
            np.random.seed(int(seed))
            summary, starts, objects = run_ph_mh_alns_aos_on_jobs_data(sliced_ops, num_machines=m_val, magazine_capacity=c_val, case_label=f"6M140_n{n_slice}", seed=int(seed), mh_time_seconds=mh_time_seconds, mh_no_improvement_limit=mh_no_improvement_limit, mh_pop_size=mh_pop_size, mh_max_generations=mh_max_generations, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, verbose=verbose, show_progress=show_progress)
            summary = _enrich_summary_with_best_alns_metadata(summary, objects)
            summary["n"] = n_slice
            for row in starts:
                row["n"] = n_slice
            run_records.append(summary)
            start_records.extend(starts)
    summary_df = _paper_like_table8_from_hybrid(run_records)
    print(summary_df.to_string(index=False))
    return summary_df, pd.DataFrame(run_records), pd.DataFrame(start_records)

def run_hybrid_table14_replications(num_runs=10, case_names=("2M38", "2M46", "6M140", "6M163"), mh_time_seconds=MAX_TIME_SECONDS, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, mh_pop_size=POP_SIZE, mh_max_generations=None, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, show_progress=True, verbose=False):
    print("\n[EXACT REPLICATION: TABLE 14 - Production Base-Case Workcenters]")
    run_records = []
    start_records = []
    case_iterator = tqdm(list(case_names), desc="Table 14 cases", disable=not show_progress)
    for case_name in case_iterator:
        case_iterator.set_postfix(case=case_name)
        case_file = resolve_kmwe_case_file(case_name)
        seed_iterator = tqdm(range(num_runs), desc=f"Seeds {case_name}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            random.seed(int(seed))
            np.random.seed(int(seed))
            summary, starts, objects = run_ph_mh_alns_aos_on_file(case_file=case_file, seed=int(seed), mh_time_seconds=mh_time_seconds, mh_no_improvement_limit=mh_no_improvement_limit, mh_pop_size=mh_pop_size, mh_max_generations=mh_max_generations, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, verbose=verbose, show_progress=show_progress)
            summary = _enrich_summary_with_best_alns_metadata(summary, objects)
            summary["BaseCase"] = case_name
            for row in starts:
                row["BaseCase"] = case_name
            run_records.append(summary)
            start_records.extend(starts)
    summary_df = _paper_like_table14_from_hybrid(run_records)
    print(summary_df.to_string(index=False))
    return summary_df, pd.DataFrame(run_records), pd.DataFrame(start_records)

def run_exact_hybrid_replications(num_runs=10, mh_time_seconds=MAX_TIME_SECONDS, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, mh_pop_size=POP_SIZE, mh_max_generations=None, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, show_progress=True, verbose=False, output_excel="hybrid_ph_mh_alns_seed_results.xlsx"):
    print(f" REPLICATING EXACT TABLES ({num_runs} SEED SAMPLES) ")
    table8_summary, table8_runs, table8_starts = run_hybrid_table8_replications(num_runs=num_runs, mh_time_seconds=mh_time_seconds, mh_no_improvement_limit=mh_no_improvement_limit, mh_pop_size=mh_pop_size, mh_max_generations=mh_max_generations, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, show_progress=show_progress, verbose=verbose)
    table14_summary, table14_runs, table14_starts = run_hybrid_table14_replications(num_runs=num_runs, mh_time_seconds=mh_time_seconds, mh_no_improvement_limit=mh_no_improvement_limit, mh_pop_size=mh_pop_size, mh_max_generations=mh_max_generations, alns_time_seconds=alns_time_seconds, alns_iterations=alns_iterations, alns_no_improvement_limit=alns_no_improvement_limit, show_progress=show_progress, verbose=verbose)
    export_seed_results_to_excel(output_excel, table8_summary=table8_summary, table8_seed_results=table8_runs, table8_alns_starts=table8_starts, table14_summary=table14_summary, table14_seed_results=table14_runs, table14_alns_starts=table14_starts)
    return {"table8_summary": table8_summary, "table8_runs": table8_runs, "table8_starts": table8_starts, "table14_summary": table14_summary, "table14_runs": table14_runs, "table14_starts": table14_starts}

def run_exact_paper_replications(num_runs=10, output_excel="hybrid_ph_mh_alns_seed_results.xlsx"):
    return run_exact_hybrid_replications(num_runs=num_runs, output_excel=output_excel)


run_exact_hybrid_replications(num_runs=10, mh_time_seconds=3600.0, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, show_progress=True)