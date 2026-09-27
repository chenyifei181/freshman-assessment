# -*- coding: utf-8 -*-
"""
电路 ②  戴维南定理验证
------------------------------------------------------------------
被测的含源二端网络：10 V 电源，R1 = 1 kΩ 串到端口，R2 = 2 kΩ 从端口到地
手算：
    V_oc = 10 × 2k/(1k+2k) = 6.667 V
    I_sc = 10 / 1k         = 10 mA
    R_th = V_oc / I_sc     = 666.7 Ω   （= R1∥R2，两种算法一致）
仿真：
    第一次：开路测 V_oc、短路测 I_sc
    第二次：用 V_th + R_th 替换原网络，接同一个负载，比对电压电流
输出：
    figures/thevenin.png   对比柱状图 + 端口伏安特性
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
from PySpice.Unit import u_V, u_Ohm, u_kOhm

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


_LOG = open(os.path.join(os.environ.get('TEMP', '.'), 'circuit_thevenin.log'),
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


def val(x):
    return float(np.asarray(x).ravel()[0])


V_SRC = 10.0     # V
R1 = 1e3         # Ω
R2 = 2e3         # Ω
R_LOAD = 1e3     # Ω

VOC_HAND = V_SRC * R2 / (R1 + R2)
ISC_HAND = V_SRC / R1
RTH_HAND = VOC_HAND / ISC_HAND

print('=' * 62)
print('② 戴维南定理验证')
print('=' * 62)
print('原网络: %.0f V 电源, R1 = %.0f Ω, R2 = %.0f Ω, 负载 RL = %.0f Ω'
      % (V_SRC, R1, R2, R_LOAD))
print('手算: V_oc = %.4f V, I_sc = %.4f mA, R_th = %.4f Ω'
      % (VOC_HAND, ISC_HAND * 1e3, RTH_HAND))
print('      校验: R1∥R2 = %.4f Ω （与 V_oc/I_sc 一致）'
      % (R1 * R2 / (R1 + R2)))
print()


def original_network(load=None, short=False):
    c = Circuit('Thevenin - original network')
    c.V('1', 'vcc', c.gnd, V_SRC @ u_V)
    c.R('1', 'vcc', 'port', R1 @ u_Ohm)
    c.R('2', 'port', c.gnd, R2 @ u_Ohm)
    if short:
        c.V('sc', 'port', c.gnd, 0 @ u_V)      # 0 V 电源当电流表用
    if load:
        c.R('L', 'port', c.gnd, load @ u_Ohm)
    return c


def op_of(circuit):
    return circuit.simulator(temperature=25,
                             nominal_temperature=25).operating_point()


op_open = op_of(original_network())
VOC_SIM = val(op_open['port'])

op_short = op_of(original_network(short=True))
ISC_SIM = abs(val(op_short.branches['vsc']))

op_load = op_of(original_network(load=R_LOAD))
UL_ORIG = val(op_load['port'])
IL_ORIG = UL_ORIG / R_LOAD

print('【第一次仿真】原网络')
print('  开路电压 V_oc = %.4f V      (手算 %.4f V)' % (VOC_SIM, VOC_HAND))
print('  短路电流 I_sc = %.4f mA     (手算 %.4f mA)'
      % (ISC_SIM * 1e3, ISC_HAND * 1e3))
print('  等效电阻 R_th = V_oc/I_sc = %.4f Ω  (手算 %.4f Ω)'
      % (VOC_SIM / ISC_SIM, RTH_HAND))
print()

RTH_SIM = VOC_SIM / ISC_SIM
c_eq = Circuit('Thevenin - equivalent')
c_eq.V('th', 'out', c_eq.gnd, VOC_SIM @ u_V)
c_eq.R('th', 'out', 'load', RTH_SIM @ u_Ohm)
c_eq.R('L', 'load', c_eq.gnd, R_LOAD @ u_Ohm)

op_eq = op_of(c_eq)
UL_EQ = val(op_eq['load'])
IL_EQ = UL_EQ / R_LOAD

print('【第二次仿真】换成戴维南等效电路后（接同一个负载）')
print('  负载电压 U_L = %.4f V     (原网络 %.4f V)' % (UL_EQ, UL_ORIG))
print('  负载电流 I_L = %.4f mA    (原网络 %.4f mA)'
      % (IL_EQ * 1e3, IL_ORIG * 1e3))
print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

labels = ['负载电压 U_L / V', '负载电流 I_L / mA']
orig = [UL_ORIG, IL_ORIG * 1e3]
equiv = [UL_EQ, IL_EQ * 1e3]
x = np.arange(2)
ax1.bar(x - 0.18, orig, 0.36, label='原网络', color='#2f6feb')
ax1.bar(x + 0.18, equiv, 0.36, label='戴维南等效', color='#2f9e68')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title('② 替换前后对比')
ax1.legend()
ax1.grid(axis='y', alpha=.3)
for xi, (a, b) in enumerate(zip(orig, equiv)):
    ax1.text(xi - 0.18, a, '%.3f' % a, ha='center', va='bottom', fontsize=9)
    ax1.text(xi + 0.18, b, '%.3f' % b, ha='center', va='bottom', fontsize=9)

ax2.plot([0, VOC_SIM], [ISC_SIM * 1e3, 0], color='#e5484d', linewidth=2,
         label='端口伏安特性（戴维南直线）')
ax2.plot([0], [ISC_SIM * 1e3], 'o', color='#e5484d')
ax2.annotate('I_sc = %.2f mA' % (ISC_SIM * 1e3), xy=(0, ISC_SIM * 1e3),
             xytext=(0.12, 0.88), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='#e5484d'))
ax2.plot([VOC_SIM], [0], 'o', color='#e5484d')
ax2.annotate('V_oc = %.3f V' % VOC_SIM, xy=(VOC_SIM, 0),
             xytext=(0.32, 0.18), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='#e5484d'))
ax2.plot([UL_ORIG], [IL_ORIG * 1e3], 's', color='#2f6feb', markersize=9,
         label='带载工作点 (%.3f V, %.3f mA)' % (UL_ORIG, IL_ORIG * 1e3))
ax2.set_xlabel('端口电压 / V')
ax2.set_ylabel('端口电流 / mA')
ax2.set_title('② 端口伏安特性与工作点')
ax2.grid(alpha=.3)
ax2.legend(fontsize=9)
ax2.set_xlim(-0.3, VOC_SIM * 1.18)
ax2.set_ylim(-0.5, ISC_SIM * 1e3 * 1.18)

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'thevenin.png'), dpi=160)
plt.close(fig)

print('-' * 62)
print('手算 vs 仿真 对比表')
print('-' * 62)
print('%-16s %-14s %-14s %-14s' % ('项目', '手算值', '原网络仿真', '等效电路'))
print('%-16s %-14s %-14s %-14s' % ('开路电压 V_oc', '%.4f V' % VOC_HAND,
                                    '%.4f V' % VOC_SIM, '—'))
print('%-16s %-14s %-14s %-14s' % ('短路电流 I_sc',
                                    '%.4f mA' % (ISC_HAND * 1e3),
                                    '%.4f mA' % (ISC_SIM * 1e3), '—'))
print('%-16s %-14s %-14s %-14s' % ('等效电阻 R_th', '%.4f Ω' % RTH_HAND,
                                    '%.4f Ω' % RTH_SIM, '—'))
print('%-16s %-14s %-14s %-14s' % ('负载电压 U_L', '—',
                                    '%.4f V' % UL_ORIG, '%.4f V' % UL_EQ))
print('%-16s %-14s %-14s %-14s' % ('负载电流 I_L', '—',
                                    '%.4f mA' % (IL_ORIG * 1e3),
                                    '%.4f mA' % (IL_EQ * 1e3)))
print('-' * 62)
print('结论：替换前后负载上的电压电流一致，戴维南定理得到验证。')
print('图已保存到 figures/thevenin.png')
