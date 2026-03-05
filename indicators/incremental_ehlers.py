"""
CryptoLab — Incremental Ehlers CyberCycle Processor
====================================================
State-machine that processes one bar at a time, producing the exact
same numerical results as the vectorized ehlers.py indicators.

VERSION 2: Partial Bar Support
-------------------------------
Adds update_partial() which simulates how TradingView evaluates
indicators on every tick (or 1m bar) within a higher-timeframe bar.

The key insight:
  - Bars 1h[0..i-1] are fixed history.
  - Bar 1h[i] is "in construction": OHLCV changes with each new 1m candle.
  - CyberCycle is evaluated with partial bar close as the "current close".
  - A signal fires the moment the crossover condition is met mid-bar.
  - Entry price = 1m close at that moment (not the 1h close).

State management for partial bars:
  - update_partial(is_bar_close=False):
      1. Take snapshot at bar start (once per 1h bar)
      2. Compute indicators with partial OHLCV
      3. Revert all state back to snapshot
      4. If signal fired, update last_signal_bar IN the snapshot
         so subsequent 1m bars in the same 1h bar apply the throttle.
  - update_partial(is_bar_close=True):
      Same as update() — commits permanently, bar_count advances.

Usage:
    proc = IncrementalCyberCycle(strategy.params, detail_tf_ratio=60)

    for i, main_bar in enumerate(main_1h_bars):
        partial_high = partial_low = partial_vol = 0.0
        detail_1m = get_detail_bars(main_bar)
        n = len(detail_1m)

        for j, m in enumerate(detail_1m):
            if j == 0:
                partial_high, partial_low = m.high, m.low
            else:
                partial_high = max(partial_high, m.high)
                partial_low  = min(partial_low,  m.low)
            partial_vol += m.volume

            signal = proc.update_partial(
                high=partial_high,
                low=partial_low,
                close=m.close,          # ← "current tick"
                volume=partial_vol,
                timestamp=m.timestamp,
                is_bar_close=(j == n - 1),
            )
            if signal:
                open_position(signal)   # entry at m.close (intrabar price)
"""

import math
from typing import Optional, Dict, Any

from strategies.base import Signal


# ═══════════════════════════════════════════════════════════════
#  CIRCULAR BUFFER
# ═══════════════════════════════════════════════════════════════

class _RingBuf:
    __slots__ = ('_buf', '_cap', '_idx', '_count')

    def __init__(self, capacity: int, fill: float = 0.0):
        self._buf   = [fill] * capacity
        self._cap   = capacity
        self._idx   = 0
        self._count = 0

    def push(self, value: float):
        self._buf[self._idx % self._cap] = value
        self._idx  += 1
        if self._count < self._cap:
            self._count += 1

    def ago(self, n: int) -> float:
        if n >= self._count:
            return 0.0
        return self._buf[(self._idx - 1 - n) % self._cap]

    @property
    def last(self) -> float:
        return self.ago(0)

    def min_max(self) -> tuple:
        if self._count == 0:
            return (0.0, 0.0)
        a = self._buf[:self._count] if self._count < self._cap else self._buf
        return (min(a), max(a))

    @property
    def count(self) -> int:
        return self._count

    def snapshot(self):
        return (list(self._buf), self._idx, self._count)

    def restore(self, s):
        self._buf, self._idx, self._count = list(s[0]), s[1], s[2]


# ═══════════════════════════════════════════════════════════════
#  ALPHA COMPUTERS
#  update()         → commits state
#  update_partial() → computes then reverts state  (snapshot/restore)
# ═══════════════════════════════════════════════════════════════

class _ManualAlpha:
    def __init__(self, alpha):
        self._a = alpha
        self._p = max(2.0, (2.0 / alpha) - 1.0)
    def update(self, hl2):         return self._a, self._p
    def update_partial(self, hl2): return self._a, self._p   # stateless


class _KalmanAlpha:
    def __init__(self, pn, mn, af, al, sens):
        self.pn, self.mn, self.af, self.al, self.sens = pn, mn, af, al, sens
        self.k_x = 0.0; self.k_P = 1.0; self.innov_ema = 0.001
        self._first = True

    def _run(self, hl2):
        if self._first:
            self.k_x = hl2; self._first = False
        P_pred  = self.k_P + self.pn
        innov   = hl2 - self.k_x
        S       = P_pred + self.mn
        K       = P_pred / S if S > 1e-12 else 0.5
        self.k_x = self.k_x + K * innov
        self.k_P = (1 - K) * P_pred
        self.innov_ema = 0.05*abs(innov) + 0.95*self.innov_ema
        norm = abs(innov) / self.innov_ema if self.innov_ema > 1e-12 else 1.0
        sig  = 1 / (1 + math.exp(-((norm - 1) * self.sens)))
        a    = max(self.al, min(self.af, self.al + (self.af - self.al)*sig))
        return a, max(2.0, (2.0/a) - 1.0)

    def update(self, hl2): return self._run(hl2)
    def update_partial(self, hl2):
        kx, kP, ie, f = self.k_x, self.k_P, self.innov_ema, self._first
        r = self._run(hl2)
        self.k_x, self.k_P, self.innov_ema, self._first = kx, kP, ie, f
        return r


class _HomodyneAlpha:
    def __init__(self, min_p=3.0, max_p=40.0):
        self.min_p, self.max_p = min_p, max_p
        self.sb = _RingBuf(8); self.db = _RingBuf(8)
        self.q1 = _RingBuf(8); self.i1 = _RingBuf(8); self.sr = _RingBuf(4)
        self.I2 = self.Q2 = self.Re = self.Im = 0.0
        self.hdp = self.smp = 15.0

    def _h(self, buf, adj):
        return (0.0962*buf.ago(0) + 0.5769*buf.ago(2)
                - 0.5769*buf.ago(4) - 0.0962*buf.ago(6)) * adj

    def _snap(self):
        return (self.sb.snapshot(), self.db.snapshot(),
                self.q1.snapshot(), self.i1.snapshot(), self.sr.snapshot(),
                self.I2, self.Q2, self.Re, self.Im, self.hdp, self.smp)

    def _rest(self, s):
        sb,db,q1,i1,sr,I2,Q2,Re,Im,hdp,smp = s
        self.sb.restore(sb); self.db.restore(db)
        self.q1.restore(q1); self.i1.restore(i1); self.sr.restore(sr)
        self.I2,self.Q2,self.Re,self.Im,self.hdp,self.smp = I2,Q2,Re,Im,hdp,smp

    def _run(self, hl2):
        self.sr.push(hl2)
        s = [self.sr.ago(k) for k in range(4)]
        sm = (4*s[0]+3*s[1]+2*s[2]+s[3])/10; self.sb.push(sm)
        adj = 0.075*self.hdp + 0.54
        det = self._h(self.sb, adj); self.db.push(det)
        q1v = self._h(self.db, adj); self.q1.push(q1v)
        i1v = self.db.ago(3);        self.i1.push(i1v)
        jI  = self._h(self.i1, adj); jQ = self._h(self.q1, adj)
        I2 = 0.2*(i1v-jQ)+0.8*self.I2; Q2 = 0.2*(q1v+jI)+0.8*self.Q2
        Re = 0.2*(I2*self.I2+Q2*self.Q2)+0.8*self.Re
        Im = 0.2*(I2*self.Q2-Q2*self.I2)+0.8*self.Im
        pa = math.atan(Im/Re) if abs(Im)>1e-10 and abs(Re)>1e-10 else 0.0
        rp = (2*math.pi/pa) if pa > 0.001 else self.hdp
        rp = max(self.min_p, min(self.max_p,
               max(0.67*self.hdp, min(1.5*self.hdp, rp))))
        self.hdp = 0.2*rp + 0.8*self.hdp
        self.smp = max(self.min_p, min(self.max_p,
                       0.33*self.hdp + 0.67*self.smp))
        self.I2,self.Q2,self.Re,self.Im = I2,Q2,Re,Im
        a = 2.0/(self.smp+1.0); return a, self.smp

    def update(self, hl2): return self._run(hl2)
    def update_partial(self, hl2):
        s=self._snap(); r=self._run(hl2); self._rest(s); return r


class _MamaAlpha:
    def __init__(self, fl=0.5, sl=0.05):
        self.fl, self.sl = fl, sl
        self.sr=_RingBuf(4); self.sm=_RingBuf(8)
        self.db=_RingBuf(8); self.q1=_RingBuf(8); self.i1=_RingBuf(8)
        self.I2=self.Q2=0.0; self.pp=0.0; self.dps=5.0; self.adj=1.665

    def _h(self, b):
        return (0.0962*b.ago(0)+0.5769*b.ago(2)
                -0.5769*b.ago(4)-0.0962*b.ago(6))*self.adj

    def _snap(self):
        return (self.sr.snapshot(),self.sm.snapshot(),self.db.snapshot(),
                self.q1.snapshot(),self.i1.snapshot(),
                self.I2,self.Q2,self.pp,self.dps)

    def _rest(self, s):
        sr,sm,db,q1,i1,I2,Q2,pp,dps = s
        self.sr.restore(sr); self.sm.restore(sm); self.db.restore(db)
        self.q1.restore(q1); self.i1.restore(i1)
        self.I2,self.Q2,self.pp,self.dps = I2,Q2,pp,dps

    def _run(self, hl2):
        self.sr.push(hl2)
        s=[self.sr.ago(k) for k in range(4)]
        sm=(4*s[0]+3*s[1]+2*s[2]+s[3])/10; self.sm.push(sm)
        det=self._h(self.sm); self.db.push(det)
        q1v=self._h(self.db); self.q1.push(q1v)
        i1v=self.db.ago(3);   self.i1.push(i1v)
        jI=self._h(self.i1);  jQ=self._h(self.q1)
        I2=0.2*(i1v-jQ)+0.8*self.I2; Q2=0.2*(q1v+jI)+0.8*self.Q2
        sI=i1v+self.db.ago(4); sQ=q1v+self.q1.ago(1)
        rp = math.atan(abs(sQ/sI))*(180/math.pi) if abs(sI)>0.001 else 90.0
        if sI<0 and sQ>0: rp=180-rp
        elif sI<0 and sQ<0: rp=180+rp
        elif sI>0 and sQ<0: rp=360-rp
        dp=max(1.0, min(60.0, self.pp-rp))
        self.dps=max(1.0, min(60.0, 0.33*dp+0.67*self.dps))
        a=max(self.sl, min(self.fl, self.fl/self.dps))
        self.pp=rp; self.I2,self.Q2=I2,Q2
        return a, max(2.0, (2.0/a)-1.0)

    def update(self, hl2): return self._run(hl2)
    def update_partial(self, hl2):
        s=self._snap(); r=self._run(hl2); self._rest(s); return r


class _AutocorrelationAlpha:
    def __init__(self, min_p=6, max_p=48, avg_len=3):
        self.min_p,self.max_p,self.avg_len = min_p,max_p,avg_len
        self.step = max(1,(max_p-min_p)//10)
        a1=(0.707*2*math.pi)/max_p
        self.ahp=(math.cos(a1)+math.sin(a1)-1)/math.cos(a1)
        a1s=math.exp(-1.414*math.pi/min_p)
        b1s=2*a1s*math.cos(1.414*math.pi/min_p)
        self.c2=b1s; self.c3=-a1s*a1s; self.c1=1-self.c2-self.c3
        self.hp=_RingBuf(3); self.sr=_RingBuf(3)
        self.fr=_RingBuf(3); self.fh=_RingBuf(max(max_p*avg_len+20,250))
        self.bp=15.0; self.bc=0; self.warmup=max_p*avg_len+10

    def _snap(self):
        return (self.hp.snapshot(),self.sr.snapshot(),
                self.fr.snapshot(),self.fh.snapshot(),self.bp,self.bc)

    def _rest(self, s):
        hp,sr,fr,fh,bp,bc = s
        self.hp.restore(hp); self.sr.restore(sr)
        self.fr.restore(fr); self.fh.restore(fh)
        self.bp,self.bc = bp,bc

    def _run(self, hl2):
        self.bc += 1
        ah=self.ahp
        s0=hl2; s1=self.sr.ago(0); s2=self.sr.ago(1); self.sr.push(s0)
        h1=self.hp.ago(0); h2=self.hp.ago(1)
        hp=((1-ah/2)**2*(s0-2*s1+s2)+2*(1-ah)*h1-(1-ah)**2*h2)
        self.hp.push(hp)
        f1=self.fr.ago(0); f2=self.fr.ago(1); hp1=self.hp.ago(1)
        filt=self.c1*(hp+hp1)/2+self.c2*f1+self.c3*f2
        self.fr.push(filt); self.fh.push(filt)
        if self.bc < self.warmup:
            a=2.0/(self.bp+1.0); return a, self.bp
        best,mx = self.bp, 0.0
        for p in range(self.min_p, self.max_p+1, self.step):
            sx=sy=sxx=syy=sxy=0.0; cnt=min(self.avg_len*p,200)
            for j in range(cnt):
                x=self.fh.ago(j); y=self.fh.ago(j+p)
                sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y
            d=(cnt*sxx-sx*sx)*(cnt*syy-sy*sy)
            c=(cnt*sxy-sx*sy)/math.sqrt(d) if d>0 else 0.0
            if c>mx: mx=c; best=float(p)
        self.bp=max(self.min_p,min(self.max_p,0.25*best+0.75*self.bp))
        a=2.0/(self.bp+1.0); return a,self.bp

    def update(self, hl2): return self._run(hl2)
    def update_partial(self, hl2):
        s=self._snap(); r=self._run(hl2); self._rest(s); return r


# ═══════════════════════════════════════════════════════════════
#  MAIN PROCESSOR
# ═══════════════════════════════════════════════════════════════

class IncrementalCyberCycle:
    """
    Incremental CyberCycle with partial bar support.

    Bar-close mode:   update(high, low, close, volume, timestamp)
    Partial bar mode: update_partial(..., is_bar_close=True/False)

    See module docstring for full usage example.
    """

    def __init__(self, params: Dict[str, Any], detail_tf_ratio: int = 1):
        self.p        = dict(params)
        self._tf_ratio = max(1, int(detail_tf_ratio))
        self._build_alpha()
        self.reset()

    def _build_alpha(self):
        m = self.p.get('alpha_method', 'kalman')
        if   m == 'manual':
            self._alpha = _ManualAlpha(self.p.get('manual_alpha', 0.35))
        elif m == 'kalman':
            self._alpha = _KalmanAlpha(
                self.p.get('kal_process_noise', 0.01),
                self.p.get('kal_meas_noise',    0.5),
                self.p.get('kal_alpha_fast',     0.5),
                self.p.get('kal_alpha_slow',     0.05),
                self.p.get('kal_sensitivity',    2.0),
            )
        elif m == 'homodyne':
            self._alpha = _HomodyneAlpha(
                self.p.get('hd_min_period', 3.0),
                self.p.get('hd_max_period', 40.0),
            )
        elif m == 'mama':
            self._alpha = _MamaAlpha(
                self.p.get('mama_fast', 0.5),
                self.p.get('mama_slow', 0.05),
            )
        elif m == 'autocorrelation':
            self._alpha = _AutocorrelationAlpha(
                self.p.get('ac_min_period',  6),
                self.p.get('ac_max_period',  48),
                self.p.get('ac_avg_length',  3),
            )
        else:
            self._alpha = _ManualAlpha(0.35)

    def reset(self):
        self._build_alpha()
        self._src   = _RingBuf(4);  self._sm = _RingBuf(3)
        self._cyc   = _RingBuf(4);  self._cw = _RingBuf(10)
        self._it    = _RingBuf(3);  self._cl = _RingBuf(3)
        self._vb    = _RingBuf(20); self._ab = _RingBuf(max(20, self.p.get('cycle_str_lookback', 50)))
        self._trig  = 0.0;  self._tk = 2.0/(self.p.get('trigger_ema', 14)+1.0)
        self._ti    = False
        self._ita   = self.p.get('itrend_alpha', 0.07)
        self._fish  = 0.0
        self._atr   = 0.0;  self._atrc = 0;  self._pc = 0.0
        self._vs    = 0.0
        self._htfs  = 0.0;  self._htfc = 0.0; self._htk = 2.0/41.0
        self._pcy   = 0.0;  self._ptr  = 0.0
        self._csi   = False
        self._bar   = 0;    self._lsb  = -9999
        self._dc    = 0;    self._cd   = -1
        # Partial bar state
        self._popen = False
        self._bsnap = None

    # ── Snapshot ──
    def _snap(self):
        return {
            'src': self._src.snapshot(), 'sm':  self._sm.snapshot(),
            'cyc': self._cyc.snapshot(), 'cw':  self._cw.snapshot(),
            'it':  self._it.snapshot(),  'cl':  self._cl.snapshot(),
            'vb':  self._vb.snapshot(),  'ab':  self._ab.snapshot(),
            'trig':self._trig, 'ti':self._ti, 'fish':self._fish,
            'atr': self._atr, 'atrc':self._atrc, 'pc':self._pc,
            'vs':  self._vs, 'htfs':self._htfs, 'htfc':self._htfc,
            'pcy': self._pcy, 'ptr':self._ptr, 'csi':self._csi,
            'bar': self._bar, 'lsb':self._lsb, 'dc':self._dc, 'cd':self._cd,
        }

    def _rest(self, s):
        self._src.restore(s['src']); self._sm.restore(s['sm'])
        self._cyc.restore(s['cyc']); self._cw.restore(s['cw'])
        self._it.restore(s['it']);   self._cl.restore(s['cl'])
        self._vb.restore(s['vb']);   self._ab.restore(s['ab'])
        self._trig=s['trig']; self._ti=s['ti'];   self._fish=s['fish']
        self._atr=s['atr'];   self._atrc=s['atrc']; self._pc=s['pc']
        self._vs=s['vs'];     self._htfs=s['htfs']; self._htfc=s['htfc']
        self._pcy=s['pcy'];   self._ptr=s['ptr'];   self._csi=s['csi']
        self._bar=s['bar'];   self._lsb=s['lsb']
        self._dc=s['dc'];     self._cd=s['cd']

    # ── Core computation ──
    def _compute(self, high, low, close, volume, ts, commit) -> Optional[Signal]:
        i   = self._bar
        hl2 = (high + low) / 2.0

        # 1. Alpha
        alpha, period = (self._alpha.update(hl2) if commit
                         else self._alpha.update_partial(hl2))
        floor = self.p.get('alpha_floor', 0.0)
        if floor > 0: alpha = max(alpha, floor)

        # 2. CyberCycle
        self._src.push(hl2)
        s0,s1,s2,s3 = (self._src.ago(k) for k in range(4))
        sm = (s0+2*s1+2*s2+s3)/6.0; self._sm.push(sm)
        if i < 7:
            cyc = (s0-2*s1+s2)/4.0 if i >= 2 else 0.0
        else:
            a   = alpha
            m0,m1,m2 = self._sm.ago(0), self._sm.ago(1), self._sm.ago(2)
            c1, c2   = self._cyc.ago(0), self._cyc.ago(1)
            cyc = ((1-0.5*a)**2*(m0-2*m1+m2) + 2*(1-a)*c1 - (1-a)**2*c2)

        # 3. Trigger
        self._pcy = self._cyc.ago(0)
        self._ptr = self._trig
        self._cyc.push(cyc)
        k = self._tk
        self._trig = cyc if not self._ti else k*cyc+(1-k)*self._trig
        self._ti   = True

        # 4. iTrend
        self._cl.push(close)
        a = self._ita
        if i < 3:
            it = close
        else:
            c0,c1c,c2c = self._cl.ago(0),self._cl.ago(1),self._cl.ago(2)
            t1,t2 = self._it.ago(0), self._it.ago(1)
            it = ((a-a*a/4)*c0 + 0.5*a*a*c1c - (a-0.75*a*a)*c2c
                  + 2*(1-a)*t1 - (1-a)**2*t2)
        self._it.push(it)
        bull_t = it > self._it.ago(2) if i >= 2 else False
        bear_t = it < self._it.ago(2) if i >= 2 else False

        # 5. Fisher
        self._cw.push(cyc)
        fL,fH = self._cw.min_max(); fR = fH-fL
        fV = max(-0.999, min(0.999, 2*((cyc-fL)/fR-0.5) if fR else 0.0))
        f  = 0.5*math.log((1+fV)/(1-fV))
        fish = 0.5*f + 0.5*self._fish
        fu   = fish > self._fish; fd = fish < self._fish
        self._fish = fish

        # 6. ATR
        if i == 0:
            self._pc=close; self._atr=high-low
        else:
            tr = max(high-low, abs(high-self._pc), abs(low-self._pc))
            if self._atrc < 14:
                self._atr=(self._atr*self._atrc+tr)/(self._atrc+1); self._atrc+=1
            else:
                self._atr=(self._atr*13+tr)/14
        self._pc=close; atr=self._atr

        # 7. Volume
        if self._vb.count >= 20: self._vs -= self._vb.ago(19)
        self._vb.push(volume); self._vs += volume
        vsma    = self._vs/self._vb.count if self._vb.count else 1.0
        use_vol = self.p.get('use_volume', True)
        vol_ok  = (not use_vol) or (volume/vsma >= self.p.get('volume_mult', 2.0) if vsma > 0 else True)

        # 8. HTF EMA
        hk = self._htk
        self._htfs  = hk*hl2   + (1-hk)*self._htfs
        self._htfc  = hk*close + (1-hk)*self._htfc
        use_htf  = self.p.get('use_htf', False)
        htf_buy  = (not use_htf) or (self._htfs > self._htfc)
        htf_sell = (not use_htf) or (self._htfs < self._htfc)

        # 9. OB/OS
        in_ob = cyc >  self.p.get('ob_level',  1.5)
        in_os = cyc <  self.p.get('os_level', -1.5)
        mom3  = cyc - self._cyc.ago(3) if i >= 3 else 0.0

        # 10. Crossover
        bull_x = (self._pcy <= self._ptr) and (cyc > self._trig)
        bear_x = (self._pcy >= self._ptr) and (cyc < self._trig)

        if commit: self._bar += 1

        # ── Signal gates ──
        if i < 10: return None
        if not (bull_x or bear_x): return None

        min_bars = self.p.get('min_bars', 24) * self._tf_ratio
        if i - self._lsb < min_bars: return None

        is_buy    = bull_x
        use_trend = self.p.get('use_trend', True)

        # Cycle strength
        self._ab.push(abs(cyc))
        if self._ab.count >= self._ab._cap: self._csi = True
        if self._csi:
            vals = [self._ab.ago(k) for k in range(self._ab.count)]
            th   = sorted(vals)[int(len(vals)*self.p.get('cycle_str_pctile',50)/100)]
            cs   = abs(cyc) >= th
        else:
            cs = True

        # Confidence score
        conf = 0.0
        if is_buy:
            if bull_x: conf+=20
            if (bull_t if use_trend else True): conf+=20
            if in_os:  conf+=15
            if vol_ok: conf+=15
            if fu:     conf+=10
            if mom3>0: conf+=10
            if cs:     conf+=10
        else:
            if bear_x: conf+=20
            if (bear_t if use_trend else True): conf+=20
            if in_ob:  conf+=15
            if vol_ok: conf+=15
            if fd:     conf+=10
            if mom3<0: conf+=10
            if cs:     conf+=10
        conf = min(conf, 100.0)
        if conf < self.p.get('confidence_min', 80.0): return None

        # Daily cap
        max_daily = self.p.get('max_signals_per_day', 0)
        if max_daily > 0:
            day = ts // 86400000
            if day != self._cd: self._cd=day; self._dc=0
            if self._dc >= max_daily: return None
            self._dc += 1

        # Commit last_signal_bar (both modes — partial mode updates snapshot after)
        self._lsb = i

        # ── Build Signal ──
        direction = 1 if is_buy else -1
        entry = close
        if self.p.get('sltp_type','slatr_tprr') == 'sltp_fixed':
            sl_d = entry*(1-direction*self.p.get('sl_fixed_pct',1.5)/100)
            tp1  = entry*(1+direction*self.p.get('tp1_fixed_pct',2.0)/100)
            tp2  = entry*(1+direction*self.p.get('tp2_fixed_pct',4.0)/100)
            tp1s = self.p.get('tp1_fixed_size', 0.6)
            sld  = abs(entry-sl_d)
        else:
            sld  = atr * self.p.get('sl_atr_mult', 1.5)
            sl_d = entry - direction*sld
            tp1s = self.p.get('tp1_size', 0.6)
            tp1  = entry + direction*sld*self.p.get('tp1_rr', 2.0)
            tp2  = entry + direction*sld*self.p.get('tp2_rr', 3.0)
        tp2s   = round(1.0-tp1s, 8)
        be_pct = self.p.get('be_pct', 1.5)
        be_t   = entry*(1+direction*be_pct/100) if be_pct > 0 else 0.0
        utl    = self.p.get('use_trailing', True)
        tpull  = self.p.get('trail_pullback_pct', 1.0)
        tdist  = entry*(tpull/100) if utl else 0.0
        if utl:
            tact = entry*(1+direction*self.p.get('trail_activate_pct',2.0)/100)
            be_t = be_t if (be_pct>0 and abs(be_t-entry)<=abs(tact-entry)) else tact

        return Signal(
            direction=direction, confidence=conf,
            entry_price=entry, sl_price=sl_d,
            tp_levels=[tp1, tp2], tp_sizes=[tp1s, tp2s],
            leverage=self.p.get('leverage', 3.0),
            be_trigger=be_t, trailing=utl, trailing_distance=tdist,
            metadata={
                'close_on_signal': self.p.get('close_on_signal', True),
                'max_signals_per_day': max_daily,
                'alpha_method': self.p.get('alpha_method','kalman'),
                'alpha': alpha, 'period': period,
                'cycle': cyc, 'fisher': fish,
                'sl_dist_atr': sld/atr if atr>0 else 0,
                'tp1': tp1, 'tp2': tp2,
                'be_pct': be_pct, 'trail_pct': tpull if utl else 0,
                'partial_bar': not commit,
            }
        )

    # ── Public API ──

    def update(self, high: float, low: float, close: float,
               volume: float, timestamp: int) -> Optional[Signal]:
        """Bar-close mode. Identical output to vectorized ehlers.py."""
        return self._compute(high, low, close, volume, timestamp, commit=True)

    def update_partial(self, high: float, low: float, close: float,
                       volume: float, timestamp: int,
                       is_bar_close: bool = False) -> Optional[Signal]:
        """
        Partial bar mode — mirrors TradingView tick-by-tick evaluation.

        Call once per 1m bar with the CUMULATIVE 1h partial OHLCV:
          high   = max of all 1m highs seen so far in this 1h
          low    = min of all 1m lows seen so far in this 1h
          close  = close of the CURRENT 1m bar  ← "latest tick"
          volume = cumulative volume

        is_bar_close=True  → last 1m bar; commits state, bar_count advances.
        is_bar_close=False → mid-bar; state reverts after computation.

        At most ONE signal fires per 1h bar. After a signal fires, the
        last_signal_bar is persisted into the snapshot so subsequent 1m
        bars in the same 1h bar won't retrigger (min_bars throttle).
        """
        if is_bar_close:
            self._popen = False
            self._bsnap = None
            return self._compute(high, low, close, volume, timestamp, commit=True)

        # First 1m of this bar — capture clean state at bar start
        if not self._popen:
            self._bsnap = self._snap()
            self._popen = True

        # Evaluate with partial OHLCV
        signal = self._compute(high, low, close, volume, timestamp, commit=False)

        # Revert to bar-start state
        self._rest(self._bsnap)

        # If signal fired: persist last_signal_bar into snapshot so that
        # the next 1m bar in this same 1h bar sees the throttle correctly.
        if signal is not None:
            self._lsb              = self._bar   # update live state
            self._bsnap['lsb']     = self._bar   # update snapshot

        return signal

    @property
    def bar_count(self) -> int:
        return self._bar