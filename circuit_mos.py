# -*- coding: utf-8 -*-
"""
电路 ③  NMOS 共源级放大电路
------------------------------------------------------------------
给定：VDD = 5 V, Rg1 = 60 kΩ, Rg2 = 40 kΩ, Rd = 2 kΩ
      NMOS: K = 0.8 mA/V², V_th = 1 V, λ = 0.02 /V, Cb1 视为足够大
      输入 Vi = 10 mV / 1 kHz 正弦波
手算：
    V_GS = 5 × 40/(60+40)     = 2.0 V
    V_OV = V_GS − V_th        = 1.0 V
    I_D  = ½ K V_OV²          = 0.4 mA
    V_DS = 5 − 0.4 mA × 2 kΩ  = 4.2 V ≥ V_OV → 工作在饱和区
    g_m  = K V_OV             = 0.8 mA/V
    r_o  = 1/(λ I_D)          = 125 kΩ
    A_v  = −g_m (R_D ∥ r_o)   ≈ −1.575（反相放大）
说明：题目说 Cb1 足够大，即交流短路、直流开路，所以仿真里把 10 mV 正弦
      直接叠在栅极直流偏置上，这和理想耦合电容是等价的。
输出：
    figures/mos_transient.png   输入 / 输出波形（反相放大）
"""

import os
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import PySpice
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import u_V, u_mV, u_Ohm, u_kOhm, u_ms, u_us, u_kHz

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

# 把输出同时写进日志文件，这样窗口关掉也能看到结果
import sys


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for st in self._streams:
            try:
                st.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


_LOG = open(os.path.join(os.environ.get('TEMP', '.'), 'circuit_mos.log'),
            'w', encoding='utf-8')
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def setup_ngspice():
    """PySpice 自带两份 ngspice 内核，逐个试着加载，用能用的那份。"""
    base = os.path.join(os.path.dirname(PySpice.__file__), 'Spice', 'NgSpice')
    candidates = [NgSpiceShared.LIBRARY_PATH.format('')]
    roots = [base,
             os.path.join(HERE, 'tools', 'ngspice'),
             os.environ.get('NGSPICE_DIR', ''),
             r'C:\Spice64',
             r'C:\Program Files\ngspice']
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pattern in ('Spice64_dll/dll-vs/ngspice*.dll',
                        'Spice64_dll/dll/ngspice*.dll',
                        'Spice64_dll/**/ngspice*.dll',
                        '**/ngspice*.dll'):
            candidates += sorted(glob.glob(os.path.join(root, pattern),
                                           recursive=True))

    tried = set()
    for path in candidates:
        if not path or path in tried:
            continue
        tried.add(path)
        if not os.path.exists(path):
            continue
        try:
            NgSpiceShared.LIBRARY_PATH = path
            NgSpiceShared.new_instance()
            print('[ngspice] 已加载内核:', path)
            return True
        except Exception as exc:
            print('[ngspice] 加载失败:', os.path.basename(path),
                  '->', str(exc)[:90])

    print('[ngspice] 没有可用内核。PySpice 自带文件如下：')
    for folder in ('Spice64_dll/dll', 'Spice64_dll/dll-vs'):
        d = os.path.join(base, folder)
        if os.path.isdir(d):
            print('   ', folder, '->', sorted(os.listdir(d)))
    return False


NGSPICE_OK = setup_ngspice()

if not NGSPICE_OK:
    print()
    print('=' * 62)
    print('  没有检测到 ngspice 计算内核，暂时无法运行真实电路仿真')
    print('=' * 62)
    print('  PySpice 只是 Python 的调用层，真正做电路计算的是 ngspice 内核。')
    print('  安装方法见 README 的「怎么运行」一节；装好之后重新运行本脚本，')
    print('  就会打印仿真结果，并重新生成 figures 目录里的波形图。')
    print()
    print('  这个电路的静态工作点、增益手算和「手算 vs 仿真」对比表已经写在 README 里。')
    raise SystemExit(0)


def val(x):
    return float(np.asarray(x).ravel()[0])


VDD = 5.0        # V
RG1 = 60e3       # Ω
RG2 = 40e3       # Ω
RD = 2e3         # Ω
K = 0.8e-3       # A/V²
VTH = 1.0        # V
LAMBDA = 0.02    # 1/V
VI_AMP = 10e-3   # V
FREQ = 1e3       # Hz

VGS_H = VDD * RG2 / (RG1 + RG2)
VOV_H = VGS_H - VTH
ID_H = 0.5 * K * VOV_H ** 2
VDS_H = VDD - ID_H * RD
GM_H = K * VOV_H
RO_H = 1.0 / (LAMBDA * ID_H)
AV_H = -GM_H * (RD * RO_H) / (RD + RO_H)

print('=' * 62)
print('③ NMOS 共源级放大电路')
print('=' * 62)
print('手算静态工作点: V_GS = %.3f V, I_D = %.4f mA, V_DS = %.4f V'
      % (VGS_H, ID_H * 1e3, VDS_H))
print('饱和区判断: V_DS = %.3f V ≥ V_GS − V_th = %.3f V → 饱和 ✓'
      % (VDS_H, VOV_H))
print('手算小信号: g_m = %.4f mA/V, r_o = %.2f kΩ, A_v = %.4f'
      % (GM_H * 1e3, RO_H / 1e3, AV_H))
print()

circuit = Circuit('NMOS common source amplifier')
circuit.V('dd', 'vdd', circuit.gnd, VDD @ u_V)
circuit.R('g1', 'vdd', 'gatemid', RG1 @ u_Ohm)
circuit.R('g2', 'gatemid', circuit.gnd, RG2 @ u_Ohm)
circuit.SinusoidalVoltageSource('in', 'gate', 'gatemid',
                                amplitude=VI_AMP @ u_mV,
                                frequency=FREQ @ u_kHz)
circuit.R('d', 'vdd', 'out', RD @ u_Ohm)
circuit.M(1, 'out', 'gate', circuit.gnd, circuit.gnd, 'nmos',
          w=1e-6, l=1e-6)
circuit.model('nmos', 'NMOS',
              **{'level': 1, 'vto': VTH, 'kp': K, 'lambda': LAMBDA})

sim = circuit.simulator(temperature=25, nominal_temperature=25)

op = sim.operating_point()
VGS_S = val(op['gate'])
VDS_S = val(op['out'])
ID_S = (VDD - VDS_S) / RD

print('【直流工作点仿真】')
print('  V_GS = %.4f V    (手算 %.4f V)' % (VGS_S, VGS_H))
print('  I_D  = %.4f mA   (手算 %.4f mA)' % (ID_S * 1e3, ID_H * 1e3))
print('  V_DS = %.4f V    (手算 %.4f V)' % (VDS_S, VDS_H))
print('  饱和区判断: V_DS = %.3f V ≥ V_OV = %.3f V → 饱和 ✓'
      % (VDS_S, VOV_H))
print()

tran = sim.transient(step_time=2 @ u_us, end_time=4 @ u_ms)
t = np.asarray(tran.time).ravel()
vin_t = np.asarray(tran['gate']).ravel() - VGS_S      # 去掉直流分量
vout_t = np.asarray(tran['out']).ravel()

mask = t > 3e-3          # 取最后一个周期，避开启动过程
vin_pp = vin_t[mask].max() - vin_t[mask].min()
vout_pp = vout_t[mask].max() - vout_t[mask].min()
AV_S = -vout_pp / vin_pp        # 输出与输入反相

print('【瞬态仿真】输入 %.0f mV / %.0f Hz 正弦波' % (VI_AMP * 1e3, FREQ))
print('  输入峰峰值 = %.3f mV' % (vin_pp * 1e3))
print('  输出峰峰值 = %.3f mV' % (vout_pp * 1e3))
print('  实测电压增益 A_v = %.4f    (手算 %.4f)' % (AV_S, AV_H))
print('  误差 %.2f%%' % (abs(AV_S - AV_H) / abs(AV_H) * 100))
print()

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t * 1e3, vin_t * 1e3, color='#2f6feb', linewidth=2)
ax[0].set_ylabel('输入 / mV')
ax[0].set_title('③ NMOS 共源放大：输入 10 mV / 1 kHz 正弦波（已去掉直流分量）')
ax[0].grid(alpha=.3)
ax[1].plot(t * 1e3, vout_t, color='#e5484d', linewidth=2)
ax[1].set_xlabel('时间 / ms')
ax[1].set_ylabel('输出 / V')
ax[1].set_title('输出波形（围绕 V_DS = %.2f V 摆动，与输入反相）' % VDS_S)
ax[1].grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'mos_transient.png'), dpi=160)
plt.close(fig)

print('-' * 62)
print('手算 vs 仿真 对比表')
print('-' * 62)
print('%-12s %-16s %-16s %-10s' % ('项目', '手算值', '仿真值', '误差'))


def row(name, hand_txt, sim_txt, err_txt):
    print('%-12s %-16s %-16s %-10s' % (name, hand_txt, sim_txt, err_txt))


def pct(hand, simv):
    return '%.2f%%' % (abs(simv - hand) / abs(hand) * 100)


row('V_GS', '%.4f V' % VGS_H, '%.4f V' % VGS_S, pct(VGS_H, VGS_S))
row('I_D', '%.4f mA' % (ID_H * 1e3), '%.4f mA' % (ID_S * 1e3),
    pct(ID_H, ID_S))
row('V_DS', '%.4f V' % VDS_H, '%.4f V' % VDS_S, pct(VDS_H, VDS_S))
row('饱和区', '是（%.2f V ≥ %.2f V）' % (VDS_H, VOV_H),
    '是（%.2f V ≥ %.2f V）' % (VDS_S, VOV_H), '—')
row('g_m', '%.4f mA/V' % (GM_H * 1e3), '%.4f mA/V' % (2 * ID_S / VOV_H * 1e3),
    '—')
row('A_v', '%.4f' % AV_H, '%.4f' % AV_S, pct(AV_H, AV_S))
print('-' * 62)
print('波形图已保存到 figures/mos_transient.png')
