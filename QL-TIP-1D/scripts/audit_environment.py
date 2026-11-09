"""Record only task-relevant local software and hardware, without credentials."""
from pathlib import Path
import ctypes
import datetime
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import winreg

ROOT = Path(__file__).resolve().parents[1]

class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in ['total_phys', 'avail_phys', 'total_page',
                                               'avail_page', 'total_virtual', 'avail_virtual', 'avail_ext']]

def installed_apps():
    found = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for base in [r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
                     r'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall']:
            try:
                with winreg.OpenKey(hive, base) as root:
                    for i in range(winreg.QueryInfoKey(root)[0]):
                        try:
                            with winreg.OpenKey(root, winreg.EnumKey(root, i)) as sub:
                                name = winreg.QueryValueEx(sub, 'DisplayName')[0]
                                if any(s in name.lower() for s in ['origin', 'mathematica', 'wolfram']):
                                    try: version = winreg.QueryValueEx(sub, 'DisplayVersion')[0]
                                    except OSError: version = None
                                    found.append(dict(name=name, version=version))
                        except OSError: pass
            except OSError: pass
    return found

def main():
    result = dict(local_date='2026-11-03', timezone='Asia/Shanghai',
                  recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), python=sys.version, executable=sys.executable,
                  logical_cpus=os.cpu_count(), packages={})
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        result['cpu_name'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    mem = MemoryStatus(); mem.length = ctypes.sizeof(mem)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        result['ram_total_bytes'] = mem.total_phys
    for package in ['numpy', 'scipy', 'pandas', 'pytest', 'pyyaml', 'cupy', 'cupy-cuda12x', 'cupy-cuda13x']:
        try: result['packages'][package] = md.version(package)
        except md.PackageNotFoundError: result['packages'][package] = None
    result['gpu_query'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap',
                                                   '--format=csv'], text=True).strip()
    result['nvcc_path'] = shutil.which('nvcc')
    result['wolframscript_path'] = shutil.which('wolframscript')
    result['installed_apps'] = installed_apps()
    result['wolfram_kernel_test'] = 'not run; independent analytic finite-well validation selected'
    result['gpu_execution_test'] = 'not run; CPU baseline first, no GPU speedup claimed'
    (ROOT / 'docs').mkdir(exist_ok=True)
    (ROOT / 'docs/environment.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'requirements-lock.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
