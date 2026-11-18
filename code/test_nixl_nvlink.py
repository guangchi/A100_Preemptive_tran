#!/usr/bin/env python3
"""Measure NIXL UCX GPU transfer from a P process to a D process."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

P_GPU = int(os.environ.get("PD_P_GPU", "6"))
D_GPU = int(os.environ.get("PD_D_GPU", "7"))


def configure_environment():
    os.environ["UCX_TLS"] = os.environ.get("NIXL_TEST_UCX_TLS", "cuda,tcp")
    net_devices = os.environ.get("NIXL_TEST_UCX_NET_DEVICES", "lo")
    if net_devices:
        os.environ["UCX_NET_DEVICES"] = net_devices
    else:
        os.environ.pop("UCX_NET_DEVICES", None)
    os.environ["UCX_LOG_LEVEL"] = os.environ.get("NIXL_TEST_UCX_LOG_LEVEL", "info")


def gpu_uuid(index):
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, timeout=10,
    ).strip()


def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def wait_for(path: Path, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {path}")


def nvlink_data_kib(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "nvlink", "-i", str(gpu), "-gt", "d"], text=True
    )
    counters = {}
    for link, direction, value in re.findall(
        r"Link\s+(\d+): Data (Tx|Rx): (\d+) KiB", output
    ):
        counters.setdefault(int(link), {})[direction] = int(value)
    if not counters or any(counts.keys() != {"Tx", "Rx"} for counts in counters.values()):
        raise RuntimeError(f"No usable NVLink Tx/Rx counters for GPU {gpu}")
    return counters


def transfer_once(agent, local_handle, remote_handle):
    handle = agent.make_prepped_xfer(
        "WRITE", local_handle, [0], remote_handle, [0], backends=["UCX"]
    )
    status = agent.transfer(handle)
    while status == "PROC":
        status = agent.check_xfer_state(handle)
    agent.release_xfer_handle(handle)
    if status != "DONE":
        raise RuntimeError(f"NIXL transfer failed with status {status!r}")


def decode_agent_metadata(value):
    return base64.b64decode(value.encode("ascii"))


def run_d(work_dir: Path, size_mib: int):
    configure_environment()
    import torch
    import nixl

    nbytes = size_mib * 1024 * 1024
    dst = torch.zeros(nbytes // 4, device="cuda:0", dtype=torch.float32)
    agent = nixl.nixl_agent("D_decode", nixl.nixl_agent_config(backends=["UCX"]))
    registration = agent.register_memory(dst, backends=["UCX"])
    write_json(work_dir / "d.json", {
        "metadata": base64.b64encode(agent.get_agent_metadata()).decode("ascii"),
        "address": dst.data_ptr(),
        "size": nbytes,
        "device": 0,
    })

    p_info = wait_for(work_dir / "p.json")
    agent.add_remote_agent(decode_agent_metadata(p_info["metadata"]))
    write_json(work_dir / "d_ready.json", {"ready": True})
    wait_for(work_dir / "done.json", timeout=300)
    torch.cuda.synchronize()
    correct = bool(torch.all(dst == 1.0).item())
    write_json(work_dir / "d_result.json", {"correct": correct})
    agent.deregister_memory(registration, ["UCX"])
    if not correct:
        raise RuntimeError("D destination buffer validation failed")


def run_p(work_dir: Path, size_mib: int):
    configure_environment()
    import torch
    import nixl

    nbytes = size_mib * 1024 * 1024
    src = torch.ones(nbytes // 4, device="cuda:0", dtype=torch.float32)
    agent = nixl.nixl_agent("P_prefill", nixl.nixl_agent_config(backends=["UCX"]))
    registration = agent.register_memory(src, backends=["UCX"])
    d_info = wait_for(work_dir / "d.json")
    remote_name = agent.add_remote_agent(decode_agent_metadata(d_info["metadata"]))
    local_desc = agent.get_xfer_descs(src)
    local_handle = agent.prep_xfer_dlist(
        "NIXL_INIT_AGENT", local_desc, backends=["UCX"]
    )
    remote_desc = agent.get_xfer_descs(
        [(d_info["address"], nbytes, d_info["device"])], mem_type="VRAM"
    )
    remote_handle = agent.prep_xfer_dlist(
        remote_name, remote_desc, backends=["UCX"]
    )
    write_json(work_dir / "p.json", {
        "metadata": base64.b64encode(agent.get_agent_metadata()).decode("ascii"),
    })
    wait_for(work_dir / "d_ready.json")

    transfer_once(agent, local_handle, remote_handle)
    torch.cuda.synchronize()
    tx_before = nvlink_data_kib(P_GPU)
    rx_before = nvlink_data_kib(D_GPU)
    start = time.perf_counter()
    for _ in range(10):
        transfer_once(agent, local_handle, remote_handle)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tx_after = nvlink_data_kib(P_GPU)
    rx_after = nvlink_data_kib(D_GPU)
    tx_kib = sum(tx_after[link]["Tx"] - counters["Tx"] for link, counters in tx_before.items())
    rx_kib = sum(rx_after[link]["Rx"] - counters["Rx"] for link, counters in rx_before.items())
    write_json(work_dir / "p_result.json", {
        "size_mib": size_mib,
        "elapsed_s": elapsed,
        "bandwidth_gbps": nbytes * 10 / elapsed / 1e9,
        f"nvlink{P_GPU}_tx_gb": tx_kib * 1024 / 1e9,
        f"nvlink{D_GPU}_rx_gb": rx_kib * 1024 / 1e9,
    })
    write_json(work_dir / "done.json", {"done": True})
    agent.deregister_memory(registration, ["UCX"])


def run_parent(sizes):
    # Run the direct P2P reference separately; the NIXL endpoints below must
    # remain in independent processes for UCX CUDA IPC to be eligible.
    if P_GPU == D_GPU:
        raise ValueError("PD_P_GPU and PD_D_GPU must be distinct")
    p_uuid, d_uuid = gpu_uuid(P_GPU), gpu_uuid(D_GPU)
    nvlink_data_kib(P_GPU)
    nvlink_data_kib(D_GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{p_uuid},{d_uuid}"
    configure_environment()
    import torch

    print("NIXL UCX two-process GPU transfer test")
    print(f"UCX_TLS={os.environ['UCX_TLS']}")
    print(f"UCX_NET_DEVICES={os.environ.get('UCX_NET_DEVICES', '<default>')}")
    print(f"UCX_LOG_LEVEL={os.environ['UCX_LOG_LEVEL']}")
    print(f"P=physical GPU{P_GPU}, D=physical GPU{D_GPU}; separate Python processes")
    print(f"CUDA peer access P→D: {torch.cuda.can_device_access_peer(0, 1)}")
    for size_mib in sizes:
        nbytes = size_mib * 1024 * 1024
        src = torch.ones(nbytes // 4, device="cuda:0", dtype=torch.float32)
        dst = torch.empty(nbytes // 4, device="cuda:1", dtype=torch.float32)
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize()
        print(f"PyTorch P2P reference {size_mib:5d} MiB: "
              f"{nbytes * 10 / (time.perf_counter() - start) / 1e9:.1f} GB/s")
        del src, dst

    print("\nNIXL WRITE (per-transfer handle creation, submission, polling, release included)")
    for size_mib in sizes:
        with tempfile.TemporaryDirectory(prefix="nixl-p2p-") as temp:
            work_dir = Path(temp)
            env_d = os.environ.copy()
            env_d["CUDA_VISIBLE_DEVICES"] = d_uuid
            env_p = os.environ.copy()
            env_p["CUDA_VISIBLE_DEVICES"] = p_uuid
            d = subprocess.Popen(
                [sys.executable, __file__, "--role", "d", "--work-dir", str(work_dir),
                 "--size-mib", str(size_mib)], env=env_d
            )
            p = None
            try:
                wait_for(work_dir / "d.json")
                p = subprocess.Popen(
                    [sys.executable, __file__, "--role", "p", "--work-dir", str(work_dir),
                     "--size-mib", str(size_mib)], env=env_p
                )
                p_rc = p.wait(timeout=360)
                if p_rc != 0:
                    raise RuntimeError(f"P process failed with exit code {p_rc}")
                result = wait_for(work_dir / "p_result.json")
                d_result = wait_for(work_dir / "d_result.json")
                print(f"{size_mib:5d} MiB: {result['bandwidth_gbps']:.2f} GB/s "
                      f"elapsed={result['elapsed_s']:.3f}s "
                      f"NVLink{P_GPU}_TX={result[f'nvlink{P_GPU}_tx_gb']:.3f}GB "
                      f"NVLink{D_GPU}_RX={result[f'nvlink{D_GPU}_rx_gb']:.3f}GB "
                      f"correct={d_result['correct']}")
            finally:
                for worker in (p, d):
                    if worker is None:
                        continue
                    if worker.poll() is None:
                        worker.terminate()
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
    print("Transport is confirmed only when UCX selects cuda_ipc and NVLink counters increase.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["p", "d"])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--size-mib", type=int)
    args = parser.parse_args()
    if args.role:
        if not args.work_dir or not args.size_mib:
            parser.error("worker roles require --work-dir and --size-mib")
        (run_p if args.role == "p" else run_d)(args.work_dir, args.size_mib)
    else:
        sizes = [int(v) for v in os.environ.get(
            "NIXL_TEST_SIZES_MIB", "16,64,256,512,1024"
        ).split(",")]
        run_parent(sizes)


if __name__ == "__main__":
    main()
