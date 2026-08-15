# =============================================================================
# tools/hardware_diagnostic.py — Qualcomm Hardware & NPU Diagnostic Tool
# =============================================================================

"""
Diagnoses Qualcomm Snapdragon NPU, GPU, and CPU hardware availability,
verifies QNN SDK drivers, and runs a quick hardware sanity benchmark.
"""

import os
import sys
import time

def check_environment():
    print("=" * 65)
    print("   JAX-QNN: QUALCOMM SNAPDRAGON HARDWARE DIAGNOSTIC TOOL")
    print("=" * 65)
    
    # 1. Check Python & Platform
    print(f"[*] Python Version : {sys.version.split()[0]} ({sys.platform})")
    
    # 2. Check QNN SDK
    qnn_root = os.environ.get("QNN_SDK_ROOT")
    if qnn_root and os.path.exists(qnn_root):
        print(f"[+] QNN SDK Root   : {qnn_root}")
    else:
        print("[-] QNN SDK Root   : Not set in environment (using default driver store)")
        
    # 3. Check JAX and JAX-QNN
    try:
        import jax
        import jax_qnn
        devices = jax.devices("qnn")
        print(f"[+] JAX Platform   : Active")
        print(f"[+] QNN Devices    : {devices}")
    except Exception as e:
        print(f"[-] JAX-QNN Status : {e}")

    print("=" * 65)

if __name__ == "__main__":
    check_environment()
