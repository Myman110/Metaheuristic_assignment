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

def apmx_t(p1, p2):
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

def pmx_c(p1, p2):
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

def pox_edd(job_vector, cx1, cx2, ops_by_job):
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

def pox_mach(job_vector, ops_by_job, num_machines, magazine_capacity):
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
    def __init__(self, jobs, nm, cap, tau=SETUP_TIME):
        self.num_m, self.C, self.tau, self.jobs = nm, cap, tau, jobs
        self.ops_by_j, self.flat_ops = {}, []
        for op in jobs:
            jid = int(op["job_id"])
            self.ops_by_j.setdefault(jid, []).append(op)
            self.flat_ops.append(jid)
        for j in self.ops_by_j:
            self.ops_by_j[j].sort(key=lambda x: x["op_id"])
        self.decoder = Dec(self.ops_by_j, nm, cap, tau)

    def run(self, max_t=MAX_TIME_SECONDS, stagn=NO_IMPROVEMENT_LIMIT, pop_sz=POP_SIZE, t_rate=TOURNAMENT_RATE, elite_r=ELITISM_RATE, p_swap=SWAP_MUTATION_PROB, p_uni=UNIFORM_MUTATION_PROB, pox_it=B, max_g=None, rec_h=True, verbose=False):
        pop = []
        ph = PracHeur(self.jobs, self.num_m, self.C, self.tau).run()
        self.decoder.eval_ind(ph)
        pop.append(ph)
        n_ops = len(self.flat_ops)
        for _ in range(pop_sz - 1):
            jv = list(self.flat_ops)
            random.shuffle(jv)
            mv = [random.randint(1, self.num_m) for _ in range(n_ops)]
            ind = Ind(jv, mv)
            self.decoder.eval_ind(ind)
            pop.append(ind)
        best_ind = min(pop, key=lambda x: x.fit)
        f_best, no_imp, q, best_imp, gen = best_ind.fit, 0, pox_it + 1, False, 0
        t0 = time.time()
        hist = []
        if rec_h:
            hist.append({"generation": 0, "runtime": 0.0, "best_fitness": float(f_best), "current_best": float(f_best), "no_improve": 0, "used_pox": False, "improved": True})
        while True:
            elap = time.time() - t0
            if elap >= max_t or no_imp >= stagn or (max_g is not None and gen >= max_g):
                break
            off = []
            use_pox = best_imp or q <= pox_it
            while len(off) < pop_sz:
                p1, p2 = self.select_parents(pop, t_rate)
                if use_pox:
                    tp1, tp2 = apmx_t(p1.jv, p2.jv)
                    tc1, tc2, cx1, cx2 = pmx_c(tp1, tp2)
                    c1j = pox_edd([p1.jv[v - 1] for v in tc1], cx1, cx2, self.ops_by_j)
                    c2j = pox_edd([p1.jv[v - 1] for v in tc2], cx1, cx2, self.ops_by_j)
                    c1m = pox_mach(c1j, self.ops_by_j, self.num_m, self.C)
                    c2m = pox_mach(c2j, self.ops_by_j, self.num_m, self.C)
                else:
                    tp1, tp2 = apmx_t(p1.jv, p2.jv)
                    tc1, tc2, _, _ = pmx_c(tp1, tp2)
                    c1j = [p1.jv[v - 1] for v in tc1]
                    c2j = [p1.jv[v - 1] for v in tc2]
                    c1m, c2m = self.two_point_crossover(p1.mv, p2.mv)
                child1, child2 = Ind(c1j, c1m), Ind(c2j, c2m)
                self.apply_mutation(child1, p_swap, p_uni)
                self.apply_mutation(child2, p_swap, p_uni)
                self.decoder.eval_ind(child1)
                self.decoder.eval_ind(child2)
                off.extend([child1, child2])
            off = off[:pop_sz]
            se = int(elite_r * pop_sz)
            elites = sorted(pop, key=lambda x: x.fit)[:se]
            n_pop = off[:]
            if se > 0:
                for idx, el in zip(random.sample(range(pop_sz), se), elites):
                    n_pop[idx] = el
            u_sigs = set()
            f_pop = []
            for ind in n_pop:
                sig = ind.sig()
                if sig not in u_sigs:
                    u_sigs.add(sig)
                    f_pop.append(ind)
                else:
                    imm_jv = list(self.flat_ops)
                    random.shuffle(imm_jv)
                    imm_mv = [random.randint(1, self.num_m) for _ in range(n_ops)]
                    imm = Ind(imm_jv, imm_mv)
                    self.decoder.eval_ind(imm)
                    f_pop.append(imm)
            pop = f_pop
            cur_best = min(pop, key=lambda x: x.fit)
            imp = cur_best.fit < f_best
            if imp:
                f_best, best_ind, best_imp, q, no_imp = cur_best.fit, cur_best, True, 1, 0
            else:
                best_imp, q, no_imp = False, q + 1, no_imp + 1
            gen += 1
            if rec_h:
                hist.append({"generation": gen, "runtime": float(time.time() - t0), "best_fitness": float(f_best), "current_best": float(cur_best.fit), "no_improve": int(no_imp), "used_pox": bool(use_pox), "improved": bool(imp)})
        best_ind.generations = gen
        best_ind.runtime = time.time() - t0
        best_ind.history = hist
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

def clone_i(ind):
    n = Ind(ind.jv, ind.mv)
    n.fit = float(getattr(ind, "fit", float("inf")))
    n.tard = float(getattr(ind, "tard", 0.0))
    n.su = int(getattr(ind, "su", 0))
    for a in ["runtime", "generations", "su_pos", "alns_iterations", "alns_runtime", "alns_stop_reason", "alns_history", "alns_destroy_weights", "alns_repair_weights", "alns_eval_cache_hits", "alns_eval_cache_misses", "source_label", "source_rank", "source_diversity"]:
        if hasattr(ind, a):
            setattr(n, a, getattr(ind, a))
    return n

def uq_sort(inds):
    uq = {}
    for ind in inds:
        sig = ind.sig()
        if sig not in uq or ind.fit < uq[sig].fit:
            uq[sig] = clone_i(ind)
    return sorted(uq.values(), key=lambda x: (x.fit, x.tard, x.su))

def sel_best(arch):
    r = uq_sort(arch)
    if not r:
        raise ValueError()
    b = clone_i(r[0])
    b.source_rank, b.source_label, b.source_diversity = 1, "GA_best", 0.0
    return [b]

class MHArch(MathGA):
    def run_arch(self, max_t=MAX_TIME_SECONDS, stagn=NO_IMPROVEMENT_LIMIT, pop_sz=POP_SIZE, t_rate=TOURNAMENT_RATE, elite_r=ELITISM_RATE, p_swap=SWAP_MUTATION_PROB, p_uni=UNIFORM_MUTATION_PROB, pox_it=B, max_g=None, rec_h=True, verbose=False, top_g=25, show_pr=False, pr_desc="GA/MH"):
        pop, arch = [], []
        ph = PracHeur(self.jobs, self.num_m, self.C, self.tau).run()
        self.decoder.eval_ind(ph)
        pop.append(ph)
        arch.append(clone_i(ph))
        n_ops = len(self.flat_ops)
        for _ in range(pop_sz - 1):
            jv = list(self.flat_ops)
            random.shuffle(jv)
            mv = [random.randint(1, self.num_m) for _ in range(n_ops)]
            ind = Ind(jv, mv)
            self.decoder.eval_ind(ind)
            pop.append(ind)
        arch.extend(clone_i(x) for x in sorted(pop, key=lambda x: x.fit)[:top_g])
        best_ind = min(pop, key=lambda x: x.fit)
        f_best, no_imp, q, best_imp, gen = best_ind.fit, 0, pox_it + 1, False, 0
        t0 = time.time()
        hist = []
        pbar = tqdm(total=max_g, desc=pr_desc, leave=False, disable=not show_pr)
        if rec_h:
            hist.append({"generation": 0, "runtime": 0.0, "best_fitness": float(f_best), "current_best": float(f_best), "no_improve": 0, "used_pox": False, "improved": True, "archive_size": len(uq_sort(arch))})
        while True:
            elap = time.time() - t0
            if elap >= max_t or no_imp >= stagn or (max_g is not None and gen >= max_g):
                break
            off = []
            use_pox = best_imp or q <= pox_it
            while len(off) < pop_sz:
                p1, p2 = self.select_parents(pop, t_rate)
                if use_pox:
                    tp1, tp2 = apmx_t(p1.jv, p2.jv)
                    tc1, tc2, cx1, cx2 = pmx_c(tp1, tp2)
                    c1j = pox_edd([p1.jv[v - 1] for v in tc1], cx1, cx2, self.ops_by_j)
                    c2j = pox_edd([p1.jv[v - 1] for v in tc2], cx1, cx2, self.ops_by_j)
                    c1m = pox_mach(c1j, self.ops_by_j, self.num_m, self.C)
                    c2m = pox_mach(c2j, self.ops_by_j, self.num_m, self.C)
                else:
                    tp1, tp2 = apmx_t(p1.jv, p2.jv)
                    tc1, tc2, _, _ = pmx_c(tp1, tp2)
                    c1j = [p1.jv[v - 1] for v in tc1]
                    c2j = [p1.jv[v - 1] for v in tc2]
                    c1m, c2m = self.two_point_crossover(p1.mv, p2.mv)
                ch1, ch2 = Ind(c1j, c1m), Ind(c2j, c2m)
                self.apply_mutation(ch1, p_swap, p_uni)
                self.apply_mutation(ch2, p_swap, p_uni)
                self.decoder.eval_ind(ch1)
                self.decoder.eval_ind(ch2)
                off.extend([ch1, ch2])
            off = off[:pop_sz]
            se = int(elite_r * pop_sz)
            elites = sorted(pop, key=lambda x: x.fit)[:se]
            n_pop = off[:]
            if se > 0:
                for idx, el in zip(random.sample(range(pop_sz), se), elites):
                    n_pop[idx] = el
            u_sigs = set()
            f_pop = []
            for ind in n_pop:
                sig = ind.sig()
                if sig not in u_sigs:
                    u_sigs.add(sig)
                    f_pop.append(ind)
                else:
                    imm_jv = list(self.flat_ops)
                    random.shuffle(imm_jv)
                    imm_mv = [random.randint(1, self.num_m) for _ in range(n_ops)]
                    imm = Ind(imm_jv, imm_mv)
                    self.decoder.eval_ind(imm)
                    f_pop.append(imm)
            pop = f_pop
            arch.extend(clone_i(x) for x in sorted(pop, key=lambda x: x.fit)[:top_g])
            cur_best = min(pop, key=lambda x: x.fit)
            imp = cur_best.fit < f_best
            if imp:
                f_best, best_ind, best_imp, q, no_imp = cur_best.fit, cur_best, True, 1, 0
            else:
                best_imp, q, no_imp = False, q + 1, no_imp + 1
            gen += 1
            if show_pr:
                pbar.update(1)
                pbar.set_postfix(best=round(float(f_best), 2), noimp=int(no_imp), arch=len(uq_sort(arch)))
            if rec_h:
                hist.append({"generation": gen, "runtime": float(time.time() - t0), "best_fitness": float(f_best), "current_best": float(cur_best.fit), "no_improve": int(no_imp), "used_pox": bool(use_pox), "improved": bool(imp), "archive_size": len(uq_sort(arch))})
        pbar.close()
        best_ind = clone_i(best_ind)
        best_ind.generations = gen
        best_ind.runtime = time.time() - t0
        best_ind.history = hist
        best_ind.ga_archive = uq_sort(arch)
        return best_ind, best_ind.ga_archive

def get_ops_j(jobs):
    o = {}
    for op in jobs:
        o.setdefault(int(op["job_id"]), []).append(op)
    for j in o:
        o[j].sort(key=lambda x: x["op_id"])
    return o

def run_hybrid_d(jobs, nm, cap, lbl="custom", is_f=False, seed=0, mh_t=MAX_TIME_SECONDS, mh_st=NO_IMPROVEMENT_LIMIT, pop_sz=POP_SIZE, max_g=None, alns_t=600.0, alns_it=2000, alns_st=500, rst_seed=True, verbose=False, show_pr=True):
    random.seed(seed)
    np.random.seed(seed)
    TL_SIZES.clear()
    TL_SIZES.update({o["tool_set"]: o["size"] for o in jobs})
    ops_by_j = get_ops_j(jobs)
    dec = Dec(ops_by_j, nm, cap, SETUP_TIME)
    t0 = time.time()
    ph = PracHeur(jobs, nm, cap).run()
    dec.eval_ind(ph)
    ph_rt = time.time() - t0
    mh = MHArch(jobs, nm, cap)
    mh_sol, arch = mh.run_arch(max_t=mh_t, stagn=mh_st, pop_sz=pop_sz, max_g=max_g, show_pr=show_pr, pr_desc=f"GA/MH {lbl} S={seed}")
    dec.eval_ind(mh_sol)
    starts = sel_best(arch)
    alns_res = []
    for idx, start_ind in enumerate(starts, start=1):
        if rst_seed:
            random.seed(seed * 1000 + idx)
            np.random.seed(seed * 1000 + idx)
        start_ind = clone_i(start_ind)
        dec.eval_ind(start_ind)
        alns_eng = Alns(jobs, nm, cap)
        alns_sol = alns_eng.run(start_ind, max_it=alns_it, max_t=alns_t, stagn=alns_st, show_pr=show_pr, pr_desc=f"ALNS {lbl} S={seed}")
        dec.eval_ind(alns_sol)
        alns_sol.start_label = getattr(start_ind, "source_label", f"GA_start_{idx}")
        alns_sol.start_rank = idx
        alns_sol.start_fit = float(start_ind.fit)
        alns_sol.start_tard = float(start_ind.tard)
        alns_sol.start_su = int(start_ind.su)
        alns_res.append(alns_sol)
    b_hy = min(alns_res, key=lambda x: (x.fit, x.tard, x.su))
    rows = []
    for sol in alns_res:
        row = {
            "seed": seed, "start_rank": sol.start_rank, "start_label": sol.start_label,
            "start_fitness": sol.start_fit, "start_tardiness": sol.start_tard, "start_setups": sol.start_su,
            "ALNS_fitness": sol.fit, "ALNS_tardiness": sol.tard, "ALNS_setups": sol.su,
            "ALNS_runtime": getattr(sol, "alns_rt", np.nan), "ALNS_iterations": getattr(sol, "alns_it", np.nan),
            "ALNS_stop_reason": getattr(sol, "alns_stop", "unknown"), "is_best_hybrid_start": sol is b_hy
        }
        if is_f:
            row["case_file"] = lbl
        else:
            row["case_label"] = lbl
        rows.append(row)
    sum_dict = {
        "seed": seed, "PH_fitness": ph.fit, "PH_tardiness": ph.tard, "PH_setups": ph.su, "PH_runtime": ph_rt,
        "MH_fitness": mh_sol.fit, "MH_tardiness": mh_sol.tard, "MH_setups": mh_sol.su, "MH_runtime": getattr(mh_sol, "runtime", np.nan),
        "MH_generations": getattr(mh_sol, "generations", np.nan), "MH_stop_reason": getattr(mh_sol, "stop_reason", "unknown"),
        "GA_archive_unique_size": len(arch), "Hybrid_fitness": b_hy.fit, "Hybrid_tardiness": b_hy.tard, "Hybrid_setups": b_hy.su,
        "ILP_final_fitness": b_hy.fit, "ILP_final_tardiness": b_hy.tard, "ILP_final_setups": b_hy.su,
        "Hybrid_ALNS_runtime": getattr(b_hy, "alns_rt", np.nan), "Hybrid_total_runtime": ph_rt + getattr(mh_sol, "runtime", 0.0) + sum(getattr(s, "alns_rt", 0.0) for s in alns_res),
        "Best_ALNS_start_label": getattr(b_hy, "start_label", "unknown"), "Best_ALNS_start_rank": getattr(b_hy, "start_rank", np.nan)
    }
    if is_f:
        sum_dict["case_file"] = lbl
    else:
        sum_dict["case_label"] = lbl
    return sum_dict, rows, {"ph_solution": ph, "mh_solution": mh_sol, "ga_starts": starts, "alns_results": alns_res, "best_hybrid": b_hy, "ga_archive": arch}

def run_ph_mh_alns_aos_on_file(case_file, seed=0, mh_time_seconds=MAX_TIME_SECONDS, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, mh_pop_size=POP_SIZE, mh_max_generations=None, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, reset_seed_per_alns_start=True, verbose=False, show_progress=True):
    ops, nm, cap = load_actual_kmwe_instance(case_file)
    return run_hybrid_d(ops, nm, cap, case_file, True, seed, mh_time_seconds, mh_no_improvement_limit, mh_pop_size, mh_max_generations, alns_time_seconds, alns_iterations, alns_no_improvement_limit, reset_seed_per_alns_start, verbose, show_progress)

def run_ph_mh_alns_aos_on_jobs_data(jobs_data, num_machines, magazine_capacity, case_label="custom_instance", seed=0, mh_time_seconds=MAX_TIME_SECONDS, mh_no_improvement_limit=NO_IMPROVEMENT_LIMIT, mh_pop_size=POP_SIZE, mh_max_generations=None, alns_time_seconds=600.0, alns_iterations=2000, alns_no_improvement_limit=500, reset_seed_per_alns_start=True, verbose=False, show_progress=True):
    return run_hybrid_d(jobs_data, num_machines, magazine_capacity, case_label, False, seed, mh_time_seconds, mh_no_improvement_limit, mh_pop_size, mh_max_generations, alns_time_seconds, alns_iterations, alns_no_improvement_limit, reset_seed_per_alns_start, verbose, show_progress)

def run_hybrid_experiments(case_files, seeds=range(10), show_progress=True, **kwargs):
    all_sums, all_starts = [], []
    for cf in tqdm(list(case_files), desc="Cases", disable=not show_progress):
        for sd in tqdm(list(seeds), desc=f"Seeds {os.path.basename(str(cf))}", leave=False, disable=not show_progress):
            sum_d, starts, _ = run_ph_mh_alns_aos_on_file(cf, int(sd), show_progress=show_progress, **kwargs)
            all_sums.append(sum_d)
            all_starts.extend(starts)
    return pd.DataFrame(all_sums), pd.DataFrame(all_starts)

def agg_rec(records, group_col):
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(group_col, sort=False):
        ph, mh, hy = g["PH_fitness"].astype(float), g["MH_fitness"].astype(float), g["Hybrid_fitness"].astype(float)
        ph_mean, mh_mean, hy_mean = float(ph.mean()), float(mh.mean()), float(hy.mean())
        best_m = min({"PH": ph_mean, "MH": mh_mean, "ALNS": hy_mean}, key=lambda k: {"PH": ph_mean, "MH": mh_mean, "ALNS": hy_mean}[k])
        rows.append({
            group_col: key,
            "PH_μ": round(ph_mean, 2), "PH_σ": round(float(ph.std(ddof=0)), 2), "PH_best": round(float(ph.min()), 2), "PH_C.T.(s)": round(float(g["PH_runtime"].mean()), 3),
            "MH_μ": round(mh_mean, 2), "MH_σ": round(float(mh.std(ddof=0)), 2), "MH_best": round(float(mh.min()), 2), "MH_C.T.(s)": round(float(g["MH_runtime"].mean()), 3),
            "MH_gen_μ": round(float(g["MH_generations"].mean()), 1),
            "ALNS_μ": round(hy_mean, 2), "ALNS_σ": round(float(hy.std(ddof=0)), 2), "ALNS_best": round(float(hy.min()), 2),
            "ALNS_C.T.(s)": round(float(g["Hybrid_total_runtime"].mean()), 3), "ALNS_phase_C.T.(s)": round(float(g["Hybrid_ALNS_runtime"].mean()), 3),
            "ALNS_it_μ": round(float(g["Hybrid_ALNS_iterations"].mean()), 1),
            "StopReasons": ",".join(sorted(set(map(str, g["MH_stop_reason"])))),
            "ALNS_StopReasons": ",".join(sorted(set(map(str, g["Hybrid_ALNS_stop_reason"])))),
            "Gap_MH_vs_PH (%)": f"{((mh_mean - ph_mean) / max(1.0, ph_mean)) * 100.0:.2f}%",
            "Gap_ALNS_vs_PH (%)": f"{((hy_mean - ph_mean) / max(1.0, ph_mean)) * 100.0:.2f}%",
            "Gap_ALNS_vs_MH (%)": f"{((hy_mean - mh_mean) / max(1.0, mh_mean)) * 100.0:.2f}%",
            "Best_Method": best_m, "Best_μ": round(ph_mean if best_m == "PH" else (mh_mean if best_m == "MH" else hy_mean), 2)
        })
    return pd.DataFrame(rows)

def tbl8_fmt(records):
    f = agg_rec(records, "n")
    if f.empty:
        return f
    cols = ["n", "PH_μ", "PH_σ", "PH_best", "PH_C.T.(s)", "MH_μ", "MH_σ", "MH_best", "MH_C.T.(s)", "MH_gen_μ", "StopReasons", "Gap_MH_vs_PH (%)", "ALNS_μ", "ALNS_σ", "ALNS_best", "ALNS_C.T.(s)", "ALNS_phase_C.T.(s)", "ALNS_it_μ", "ALNS_StopReasons", "Gap_ALNS_vs_PH (%)", "Gap_ALNS_vs_MH (%)", "Best_Method", "Best_μ"]
    return f[cols]

def tbl14_fmt(records):
    f = agg_rec(records, "BaseCase")
    if f.empty:
        return f
    cols = ["BaseCase", "PH_μ", "PH_σ", "PH_best", "PH_C.T.(s)", "MH_μ", "MH_σ", "MH_best", "MH_C.T.(s)", "MH_gen_μ", "StopReasons", "Gap_MH_vs_PH (%)", "ALNS_μ", "ALNS_σ", "ALNS_best", "ALNS_C.T.(s)", "ALNS_phase_C.T.(s)", "ALNS_it_μ", "ALNS_StopReasons", "Gap_ALNS_vs_PH (%)", "Gap_ALNS_vs_MH (%)", "Best_Method", "Best_μ"]
    return f[cols].rename(columns={"Gap_MH_vs_PH (%)": "Net_Gap_MH (%)"})

def enrich_sum(summary, objects):
    b = objects["best_hybrid"]
    summary["Hybrid_ALNS_iterations"] = getattr(b, "alns_it", np.nan)
    summary["Hybrid_ALNS_stop_reason"] = getattr(b, "alns_stop", "unknown")
    return summary

def run_t8(num_runs=10, mh_t=MAX_TIME_SECONDS, mh_st=NO_IMPROVEMENT_LIMIT, pop_sz=POP_SIZE, max_g=None, alns_t=600.0, alns_it=2000, alns_st=500, show_pr=True, verbose=False):
    cf = resolve_kmwe_case_file("6M140")
    jobs, nm, cap = load_actual_kmwe_instance(cf)
    df_sorted = pd.DataFrame(jobs).sort_values(by="r").copy()
    run_recs, start_recs = [], []
    n_vals = [15, 25, 30, 60, 90, 120, 140]
    n_iter = tqdm(n_vals, desc="T8 n-slices", disable=not show_pr)
    for n_slice in n_iter:
        n_iter.set_postfix(n=n_slice)
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        seed_iter = tqdm(range(num_runs), desc=f"Seeds n={n_slice}", leave=False, disable=not show_pr)
        for seed in seed_iter:
            sum_d, starts, objs = run_ph_mh_alns_aos_on_jobs_data(sliced_ops, nm, cap, f"6M140_n{n_slice}", seed, mh_t, mh_st, pop_sz, max_g, alns_t, alns_it, alns_st, True, verbose, show_pr)
            sum_d = enrich_sum(sum_d, objs)
            sum_d["n"] = n_slice
            for r in starts:
                r["n"] = n_slice
            run_recs.append(sum_d)
            start_recs.extend(starts)
    return tbl8_fmt(run_recs), pd.DataFrame(run_recs), pd.DataFrame(start_recs)

def run_t14(num_runs=10, cases=("2M38", "2M46", "6M140", "6M163"), mh_t=MAX_TIME_SECONDS, mh_st=NO_IMPROVEMENT_LIMIT, pop_sz=POP_SIZE, max_g=None, alns_t=600.0, alns_it=2000, alns_st=500, show_pr=True, verbose=False):
    run_recs, start_recs = [], []
    c_iter = tqdm(cases, desc="T14 cases", disable=not show_pr)
    for name in c_iter:
        c_iter.set_postfix(case=name)
        cf = resolve_kmwe_case_file(name)
        seed_iter = tqdm(range(num_runs), desc=f"Seeds {name}", leave=False, disable=not show_pr)
        for seed in seed_iter:
            sum_d, starts, objs = run_ph_mh_alns_aos_on_file(cf, seed, mh_t, mh_st, pop_sz, max_g, alns_t, alns_it, alns_st, True, verbose, show_pr)
            sum_d = enrich_sum(sum_d, objs)
            sum_d["BaseCase"] = name
            for r in starts:
                r["BaseCase"] = name
            run_recs.append(sum_d)
            start_recs.extend(starts)
    return tbl14_fmt(run_recs), pd.DataFrame(run_recs), pd.DataFrame(start_recs)

def run_reps(num_runs=10, mh_t=MAX_TIME_SECONDS, mh_st=NO_IMPROVEMENT_LIMIT, pop_sz=POP_SIZE, max_g=None, alns_t=600.0, alns_it=2000, alns_st=500, show_pr=True, verbose=False, out_xls="hybrid_ph_mh_alns_seed_results.xlsx"):
    t8_s, t8_r, t8_st = run_t8(num_runs, mh_t, mh_st, pop_sz, max_g, alns_t, alns_it, alns_st, show_pr, verbose)
    t14_s, t14_r, t14_st = run_t14(num_runs, ("2M38", "2M46", "6M140", "6M163"), mh_t, mh_st, pop_sz, max_g, alns_t, alns_it, alns_st, show_pr, verbose)
    export_seed_results_to_excel(out_xls, table8_summary=t8_s, table8_seed_results=t8_r, table8_alns_starts=t8_st, table14_summary=t14_s, table14_seed_results=t14_r, table14_alns_starts=t14_st)
    return {"table8_summary": t8_s, "table8_runs": t8_r, "table8_starts": t8_st, "table14_summary": t14_s, "table14_runs": t14_r, "table14_starts": t14_st}

run_reps(num_runs=10, mh_t=3600.0, mh_st=NO_IMPROVEMENT_LIMIT, alns_t=600.0, alns_it=2000, alns_st=500, show_pr=True)