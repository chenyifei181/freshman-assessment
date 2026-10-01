# -*- coding: utf-8 -*-
"""
电路 ①  RC 低通滤波电路
------------------------------------------------------------------
手算：
    τ  = R × C = 1 kΩ × 100 nF = 1e-4 s = 0.1 ms
    fc = 1 / (2πRC) ≈ 1591.5 Hz ≈ 1.59 kHz
仿真：
    瞬态：输入 1 kHz 方波，观察电容充放电（输出被"抹圆"）
    交流：扫频画波特图，从曲线上找增益下降 3 dB 的频率，与 fc 对比
输出：
    figures/rc_transient.png   方波输入 / 输出瞬态波形
    figures/rc_bode.png        波特图
"""

import os
import math
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import PySpice
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import u_V, u_ms, u_us, u_ns, u_s, u_kOhm, u_nF, u_Hz, u_MHz

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


_LOG = open(os.path.join(os.environ.get('TEMP', '.'), 'circuit_rc.log'),
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
    print('  这个电路的手算过程和「手算 vs 仿真」对比表已经写在 README 里。')
    raise SystemExit(0)


def wave(analysis, node):
    return np.asarray(analysis[node]).ravel()


R = 1e3        # 1 kΩ
C = 100e-9     # 100 nF

tau_hand = R * C
fc_hand = 1.0 / (2 * math.pi * R * C)

print('=' * 62)
print('① RC 低通滤波电路')
print('=' * 62)
print('参数: R = %.0f Ω, C = %.3g F' % (R, C))
print('手算: τ = %.6g s = %.3f ms' % (tau_hand, tau_hand * 1e3))
print('      fc = %.1f Hz' % fc_hand)
print()

# ---------------- 瞬态分析 ----------------
square = Circuit('RC low pass - transient')
square.PulseVoltageSource(
    '1', 'vin', square.gnd,
    initial_value=0 @ u_V, pulsed_value=1 @ u_V,
    pulse_width=0.5 @ u_ms, period=1 @ u_ms,
    delay_time=0 @ u_s, rise_time=1 @ u_ns, fall_time=1 @ u_ns,
)
square.R(1, 'vin', 'vout', R @ u_kOhm)
square.C(1, 'vout', square.gnd, (C * 1e9) @ u_nF)

tran = square.simulator(temperature=25, nominal_temperature=25)\
             .transient(step_time=0.5 @ u_us, end_time=3 @ u_ms)
t = np.asarray(tran.time).ravel()
vin_t = wave(tran, 'vin')
vout_t = wave(tran, 'vout')

target = 0.632
hit = np.where(vout_t >= target)[0]
tau_sim = t[hit[0]] if len(hit) else float('nan')

print('【瞬态仿真】方波 0→1 V / 1 kHz')
print('  从波形量到的 τ = %.4f ms' % (tau_sim * 1e3))
print('  手算 τ = %.4f ms，误差 %.2f%%'
      % (tau_hand * 1e3, abs(tau_sim - tau_hand) / tau_hand * 100))
print()

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t * 1e3, vin_t, color='#888888', linewidth=1.6, label='输入（方波）')
ax[0].set_ylabel('电压 / V'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title('① RC 低通滤波：方波输入的瞬态响应')
ax[1].plot(t * 1e3, vout_t, color='#2f6feb', linewidth=2.2, label='输出')
ax[1].axhline(target, color='#e5484d', linestyle='--', linewidth=1)
ax[1].annotate('63.2%%，τ = %.3f ms' % (tau_hand * 1e3),
               xy=(tau_sim * 1e3, target), xytext=(0.30, 0.70),
               textcoords='axes fraction', color='#e5484d',
               arrowprops=dict(arrowstyle='->', color='#e5484d'))
ax[1].set_xlabel('时间 / ms'); ax[1].set_ylabel('电压 / V')
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'rc_transient.png'), dpi=160)
plt.close(fig)

# ---------------- 交流分析（波特图） ----------------
ac_circuit = Circuit('RC low pass - AC')
ac_circuit.SinusoidalVoltageSource('1', 'vin', ac_circuit.gnd, amplitude=1 @ u_V)
ac_circuit.R(1, 'vin', 'vout', R @ u_kOhm)
ac_circuit.C(1, 'vout', ac_circuit.gnd, (C * 1e9) @ u_nF)

ac = ac_circuit.simulator(temperature=25, nominal_temperature=25)\
               .ac(start_frequency=10 @ u_Hz, stop_frequency=1 @ u_MHz,
                   number_of_points=500, variation='dec')

freq = np.asarray(ac.frequency).ravel()
gain_db = 20 * np.log10(np.abs(wave(ac, 'vout')))

cross = np.where(np.diff(np.sign(gain_db + 3.0103)))[0]
if len(cross):
    i = cross[0]
    f1, f2 = math.log10(freq[i]), math.log10(freq[i + 1])
    g1, g2 = gain_db[i] + 3.0103, gain_db[i + 1] + 3.0103
    fc_sim = 10 ** (f1 + (0 - g1) * (f2 - f1) / (g2 - g1))
else:
    fc_sim = float('nan')

print('【交流仿真】10 Hz ～ 1 MHz')
print('  从波特图量到的 -3 dB 频率 = %.1f Hz' % fc_sim)
print('  手算 fc = %.1f Hz，误差 %.2f%%'
      % (fc_hand, abs(fc_sim - fc_hand) / fc_hand * 100))
print()

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.semilogx(freq, gain_db, color='#2f6feb', linewidth=2.2)
ax.axhline(-3.0103, color='#e5484d', linestyle='--', linewidth=1)
ax.axvline(fc_hand, color='#2f9e68', linestyle=':', linewidth=1.6,
           label='手算 fc = %.0f Hz' % fc_hand)
ax.plot([fc_sim], [-3.0103], 'o', color='#e5484d',
        label='仿真 -3 dB = %.0f Hz' % fc_sim)
ax.set_xlabel('频率 / Hz'); ax.set_ylabel('增益 / dB')
ax.set_title('① RC 低通滤波：波特图（幅频特性）')
ax.grid(which='both', alpha=.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'rc_bode.png'), dpi=160)
plt.close(fig)

print('-' * 62)
print('手算 vs 仿真 对比表')
print('-' * 62)
print('%-12s %-16s %-16s %-8s' % ('项目', '手算值', '仿真值', '误差'))
print('%-12s %-16s %-16s %-8s' % ('时间常数 τ',
                                  '%.4f ms' % (tau_hand * 1e3),
                                  '%.4f ms' % (tau_sim * 1e3),
                                  '%.2f%%' % (abs(tau_sim - tau_hand) / tau_hand * 100)))
print('%-12s %-16s %-16s %-8s' % ('截止频率 fc',
                                  '%.1f Hz' % fc_hand,
                                  '%.1f Hz' % fc_sim,
                                  '%.2f%%' % (abs(fc_sim - fc_hand) / fc_hand * 100)))
print('-' * 62)
print('波形图已保存到 figures/：rc_transient.png、rc_bode.png')
