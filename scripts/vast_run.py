"""Run one digital_brain experiment on a rented vast.ai GPU under a hard cost cap.

    python scripts/vast_run.py quote  [--gpu H100_SXM] [--max-cost 3]
    python scripts/vast_run.py launch --machine-id M --max-hourly 2.5 --config config/X.yaml --max-cost 3 --yes
    python scripts/vast_run.py cleanup --campaign data/vast/LABEL

Safety model:
- `quote` is read-only. Only `launch` rents, and only with --yes, which the
  owner gives after approving a specific quote: a machine and a max hourly rate.
  vast.ai re-issues offer ids for the same machine between searches, so the
  launcher picks that machine's current offer, never above --max-hourly.
- The rental lifetime is capped at floor((max_cost - reserve) * 3600 / hourly).
  An independent watchdog process is armed before the rental is created. At the
  deadline it destroys every instance carrying this campaign's unique label,
  even if the controller crashed or the create response was lost.
- Destruction matches label (and instance id when known), then re-lists all
  instances to verify the instance is gone. Other instances are never touched.
- The payload is `git archive HEAD`, so a run is tied to a commit, and the
  launcher refuses to start with uncommitted changes.
- The API key stays in ~/.config/vastai and is never printed or passed on.

The watchdog runs on this machine: keep WSL running and the machine awake
until the campaign reports the instance destroyed.
"""
import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CAMPAIGNS = REPO / "data" / "vast"
IMAGE = ("vastai/base-image:cuda-12.9-mini-py312-2026-09-07@"
         "sha256:81d94a88528edf3e3975cff8b5ff627f9af0d5b8184a22e75b1bddd353d5cfc2")
DISK_GB = 40
RESERVE_USD = 0.50          # transfer, rounding and posting lag
COLLECT_MARGIN_S = 600      # stop waiting for the job this long before the deadline
REMOTE = "/opt/db"
CALIBRATION_PRESENTATIONS = 200
FORECAST_SAFETY = 1.3       # training runs a bit slower than the plasticity-free test blocks
MIN_CPU_CORES, MIN_RAM_GB, MIN_CUDA = 4, 16, 12.9


# ---------------------------------------------------------------- pure helpers

def hourly_cost(offer, disk_gb=DISK_GB):
    """Conservative hourly price: GPU/host rate plus the requested disk."""
    return offer["dph_total"] + disk_gb * (offer.get("storage_cost") or 0) / 720


def max_lifetime_s(max_cost, hourly, reserve=RESERVE_USD):
    if max_cost <= reserve:
        raise ValueError(f"max_cost {max_cost} must exceed the reserve {reserve}")
    return math.floor((max_cost - reserve) * 3600 / hourly)


def eligible(offer):
    return (offer.get("cpu_arch", "amd64") == "amd64"
            and (offer.get("cuda_max_good") or 0) >= MIN_CUDA
            and (offer.get("cpu_cores_effective") or 0) >= MIN_CPU_CORES
            and (offer.get("cpu_ram") or 0) >= MIN_RAM_GB * 1024
            and (offer.get("disk_space") or 0) >= DISK_GB)


def pick_offer(offers, machine_id, max_hourly, exclude=()):
    """The approved machine's current offer within the approved rate, or, with
    machine_id None, the cheapest eligible offer within it (offers come sorted)."""
    return next((o for o in offers
                 if machine_id in (None, o.get("machine_id")) and hourly_cost(o) <= max_hourly
                 and o.get("machine_id") not in exclude), None)


def in_country(offer, country):
    """vast.ai geolocation is "City, CC" (city sometimes blank). None = anywhere.
    Setup downloads (Miniforge, CUDA, pip) are slow on distant hosts."""
    if country is None:
        return True
    return (offer.get("geolocation") or "").split(",")[-1].strip().upper() == country.upper()


def planned_bio_s(cfg):
    """Biological seconds in a run: pre-test + training + post-test presentations."""
    t, s = cfg["test"], cfg["stimulus"]
    n_test = t["n_orientations"] * len(t["offsets_px"]) * t["repeats"]
    n = cfg["training"]["n_presentations"] + 2 * n_test
    return n * (s["present_ms"] + s["rest_ms"]) / 1000


def forecast_run_s(calibration, full_bio_s, safety=FORECAST_SAFETY):
    """Wall seconds for the full run, from a short calibration run's metrics.json.
    Brian2 standalone writes results only at the end, so a run cut off by the
    deadline returns nothing: forecast before committing the rental to it."""
    perf = calibration["performance"]
    rtf = perf["bio_time_s"] / perf["sim_wall_s"]
    return full_bio_s / rtf * safety + (perf["compile_wall_s"] or 0), rtf


def owned(instances, label, instance_id=None):
    """Instances belonging to this campaign: label must match, and id too when known."""
    return [i for i in instances
            if i.get("label") == label and (instance_id is None or i.get("id") == instance_id)]


# ---------------------------------------------------------------- vast.ai API

_client = None


def api():
    """vast.ai SDK client. It reads the key from ~/.config/vastai/vast_api_key itself.
    The SDK returns dicts; the CLI's printed output is not reliably JSON
    (`attach ssh` prints a Python repr even with --raw)."""
    global _client
    if _client is None:
        from vastai.sdk import VastAI
        _client = VastAI()
    return _client


def search_offers(gpu, country=None):
    """gpu may be a comma-separated list, e.g. H100_SXM,H100_NVL,H100_PCIE."""
    offers = []
    for name in gpu.split(","):
        query = (f"gpu_name={name.strip()} num_gpus=1 verified=true rentable=true "
                 f"rented=false reliability>0.99")
        offers += api().search_offers(query=query, order="dph_total")
    return sorted((o for o in offers if eligible(o) and in_country(o, country)), key=hourly_cost)


def account_credit():
    """Only the credit figure leaves this function; the raw response can hold secrets."""
    return float(api().show_user()["credit"])


def list_instances():
    return api().show_instances() or []


def destroy_owned(state):
    """Destroy this campaign's instance(s) and verify absence. Returns a record."""
    label, iid = state["label"], state.get("instance_id")
    targets = owned(list_instances(), label, iid) or owned(list_instances(), label)
    for inst in targets:
        try:
            api().destroy_instance(inst["id"])
        except Exception:  # noqa: BLE001
            pass  # e.g. the watchdog got there first; absence is verified below
    for _ in range(20):
        remaining = owned(list_instances(), label)
        if not remaining:
            return {"destroyed": [i["id"] for i in targets], "verified_absent": True,
                    "at": now_iso()}
        time.sleep(15)
    return {"destroyed": [i["id"] for i in targets], "verified_absent": False,
            "remaining": [i["id"] for i in remaining], "at": now_iso()}


# ---------------------------------------------------------------- campaign state

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(campaign):
    return json.loads((campaign / "state.json").read_text())


def save_state(campaign, state):
    tmp = campaign / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(campaign / "state.json")


def log(campaign, msg):
    line = f"{now_iso()} {msg}"
    print(line, flush=True)
    with open(campaign / "controller.log", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- watchdog

def watchdog(campaign):
    state = load_state(campaign)
    (campaign / "guard_ready").write_text(now_iso())
    while True:
        state = load_state(campaign)
        if state.get("status") == "destroyed":
            return
        if time.time() >= state["deadline_epoch"]:
            record = destroy_owned(state)
            state.update(status="destroyed" if record["verified_absent"] else "destroy_unverified",
                         termination=record | {"by": "watchdog"})
            save_state(campaign, state)
            return
        time.sleep(20)


# ---------------------------------------------------------------- ssh helpers

def ssh_base(state):
    c = Path(state["campaign"])
    return ["-i", str(c / "id_ed25519"), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={c / 'known_hosts'}",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]


def ssh(state, command, timeout=600, check=True, stdin_null=False):
    cmd = ["ssh", *(["-n"] if stdin_null else []), *ssh_base(state), "-p", str(state["ssh_port"]),
           f"root@{state['ssh_host']}", command]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                         stdin=subprocess.DEVNULL)
    if check and out.returncode:
        raise RuntimeError(f"remote command failed ({out.returncode}): {command}\n{out.stderr[-1500:]}")
    return out


def scp(state, src, dst, timeout=1800):
    subprocess.run(["scp", *ssh_base(state), "-P", str(state["ssh_port"]), src, dst],
                   check=True, capture_output=True, timeout=timeout)


# ---------------------------------------------------------------- commands

def cmd_quote(args):
    offers = search_offers(args.gpu, args.country)
    credit = account_credit()
    busy = [i for i in list_instances()]
    print(f"account credit ${credit:.2f}; instances on account: {len(busy)}")
    where = f" in {args.country}" if args.country else ""
    print(f"{len(offers)} eligible {args.gpu} offers{where} (disk {DISK_GB} GB included):")
    for o in offers[:args.top]:
        h = hourly_cost(o)
        print(f"  machine {o['machine_id']} (offer {o['id']}): ${h:.3f}/h  -> lifetime cap {max_lifetime_s(args.max_cost, h) / 3600:.2f} h "
              f"at ${args.max_cost:.2f}  | {o['gpu_name']} {o['gpu_ram'] / 1024:.0f}GB  "
              f"cpus {o['cpu_cores_effective']:.0f}  ram {o['cpu_ram'] / 1024:.0f}GB  "
              f"cuda {o['cuda_max_good']}  rel {o['reliability2']:.3f}  {o.get('geolocation', '')}")


def check_preconditions(args):
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        sys.exit("refusing to launch: commit your changes first (the payload is git HEAD)")
    if not (REPO / args.config).is_file():
        sys.exit(f"no such config: {args.config}")
    others = list_instances()
    if others:
        sys.exit(f"refusing to launch: {len(others)} instance(s) already on the account "
                 "(shared account; one campaign at a time)")
    offer = pick_offer(search_offers(args.gpu, args.country), args.machine_id, args.max_hourly,
                       args.exclude_machines)
    if offer is None:
        sys.exit(f"machine {args.machine_id or 'any'} has no eligible offer at or below "
                 f"${args.max_hourly:.3f}/h; re-run quote")
    hourly = args.max_hourly  # cap the lifetime at the approved rate, not today's price
    credit = account_credit()
    if credit < args.max_cost + RESERVE_USD:
        sys.exit(f"credit ${credit:.2f} does not cover max cost ${args.max_cost:.2f} plus reserve")
    return offer, hourly, credit


def cmd_launch(args):
    offer, hourly, credit = check_preconditions(args)
    lifetime = min(max_lifetime_s(args.max_cost, hourly), args.max_hours * 3600)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout.strip()
    label = f"digital-brain-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    campaign = CAMPAIGNS / label
    campaign.mkdir(parents=True)
    plan = {"label": label, "machine_id": offer["machine_id"],
            "approved_machine": args.machine_id or "any", "offer_id": offer["id"],
            "gpu": offer["gpu_name"], "offer_hourly_usd": hourly_cost(offer),
            "max_hourly_usd": hourly, "max_cost_usd": args.max_cost, "lifetime_s": lifetime,
            "credit_before_usd": credit, "commit": commit, "config": args.config,
            "device": args.device, "image": IMAGE, "disk_gb": DISK_GB}
    (campaign / "quote.json").write_text(json.dumps(offer, indent=2))
    print(json.dumps(plan, indent=2))
    if not args.yes:
        print("dry run: pass --yes to rent (only after the owner approves this quote)")
        shutil.rmtree(campaign)
        return

    state = plan | {"campaign": str(campaign), "status": "arming",
                    "deadline_epoch": time.time() + lifetime, "created_at": now_iso()}
    save_state(campaign, state)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", label,
                    "-f", str(campaign / "id_ed25519")], check=True)
    with open(campaign / "watchdog.log", "a") as wlog:
        subprocess.Popen([sys.executable, __file__, "_watchdog", "--campaign", str(campaign)],
                         start_new_session=True, stdout=wlog, stderr=wlog, cwd=REPO)
    for _ in range(30):
        if (campaign / "guard_ready").exists():
            break
        time.sleep(1)
    else:
        sys.exit("watchdog did not arm; nothing was rented")
    log(campaign, f"watchdog armed; hard deadline in {lifetime / 3600:.2f} h")

    try:
        run_campaign(campaign, state, args)
    except BaseException as exc:  # noqa: BLE001  always clean up, then re-raise
        log(campaign, f"controller error: {exc!r}")
        raise
    finally:
        state = load_state(campaign)
        if state.get("status") != "destroyed":
            record = destroy_owned(state)
            state.update(status="destroyed" if record["verified_absent"] else "destroy_unverified",
                         termination=record | {"by": "controller"})
            save_state(campaign, state)
        log(campaign, f"termination: {state['termination']}")
        try:
            log(campaign, f"credit after: ${account_credit():.2f} (charges may post later)")
        except Exception as exc:  # noqa: BLE001
            log(campaign, f"could not read credit: {exc}")


def run_campaign(campaign, state, args):
    for attempt in range(3):
        try:
            receipt = api().create_instance(state["offer_id"], image=IMAGE, disk=DISK_GB,
                                            label=state["label"], ssh=True,
                                            cancel_unavail=True)
            state["instance_id"] = receipt["new_contract"]
            break
        except Exception as exc:  # noqa: BLE001  a lost response may still have created it
            found = owned(list_instances(), state["label"])
            if found:
                state["instance_id"] = found[0]["id"]
                break
            log(campaign, f"create attempt {attempt + 1} failed: {exc}")
            offer = pick_offer(search_offers(args.gpu, args.country), args.machine_id,
                               state["max_hourly_usd"], args.exclude_machines)
            if offer is None:
                raise RuntimeError("approved machine no longer offered within the approved rate")
            state.update(offer_id=offer["id"], machine_id=offer["machine_id"],
                         offer_hourly_usd=hourly_cost(offer))
    else:
        raise RuntimeError("create failed three times; nothing was rented")
    state["status"] = "created"
    save_state(campaign, state)
    log(campaign, f"instance {state['instance_id']} created from offer {state['offer_id']}")

    pub = (campaign / "id_ed25519.pub").read_text().strip()
    attached = api().attach_ssh(state["instance_id"], pub)
    if isinstance(attached, dict) and attached.get("success") is False:
        raise RuntimeError(f"attaching the ssh key failed: {attached.get('msg', attached)}")
    wait_for_ssh(campaign, state)

    gpu = ssh(state, "nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,"
                     "power.limit --format=csv,noheader").stdout.strip()
    log(campaign, f"GPU: {gpu}")
    (campaign / "gpu.txt").write_text(gpu + "\n")

    archive = campaign / "payload.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(archive), state["commit"]],
                   cwd=REPO, check=True)
    ssh(state, f"mkdir -p {REMOTE}/project")
    scp(state, str(archive), f"root@{state['ssh_host']}:{REMOTE}/payload.tar.gz")
    ssh(state, f"tar -xzf {REMOTE}/payload.tar.gz -C {REMOTE}/project")

    log(campaign, "setting up environment (Miniforge + CUDA 12.9 + requirements + tests)")
    setup = ssh(state, f"cd {REMOTE}/project && PREFIX={REMOTE}/miniforge "
                       f"bash scripts/vast_remote_setup.sh > {REMOTE}/setup.log 2>&1; echo $?",
                timeout=max(60, int(state["deadline_epoch"] - time.time() - COLLECT_MARGIN_S)))
    if setup.stdout.strip() != "0":
        collect(campaign, state)
        raise RuntimeError("remote setup failed; see setup.log in the campaign directory")

    device = f" --device {shlex.quote(args.device)}" if args.device else ""
    calibrate(campaign, state, args, device)
    # Fully detached (own session, no stdin/stdout tie to ssh). In the first
    # trial the launch ssh did not return until the job finished; with a
    # 600 s timeout that would have killed the controller on any long run.
    # Braces matter: `cd X && cmd &` backgrounds the whole list in a subshell that
    # keeps ssh's pipes open until the job ends.
    job = (f"cd {REMOTE}/project && {{ setsid nohup bash -c '{REMOTE}/miniforge/bin/conda run "
           f"-n digital-brain --no-capture-output python -m src.main --config "
           f"{shlex.quote(args.config)}{device} > {REMOTE}/run.log 2>&1; "
           f"echo $? > {REMOTE}/run.exit' < /dev/null > /dev/null 2>&1 & }}")
    try:
        ssh(state, job, timeout=60, stdin_null=True)
    except subprocess.TimeoutExpired:
        log(campaign, "launch ssh did not return in 60 s; the detached job keeps running")
    alive = ssh(state, f"pgrep -f '[s]rc.main --config' > /dev/null || test -f {REMOTE}/run.exit",
                check=False).returncode == 0
    if not alive:
        raise RuntimeError("job is neither running nor finished after launch")
    state["status"] = "running"
    state["job_started"] = now_iso()
    save_state(campaign, state)
    log(campaign, f"job started: {args.config}{device}")

    while time.time() < state["deadline_epoch"] - COLLECT_MARGIN_S:
        done = ssh(state, f"cat {REMOTE}/run.exit 2>/dev/null", check=False).stdout.strip()
        if done:
            log(campaign, f"job exited with status {done}")
            state["job_exit"] = int(done)
            break
        time.sleep(60)
    else:
        log(campaign, "deadline approaching: collecting partial results")
        state["job_exit"] = None
    save_state(campaign, state)
    collect(campaign, state)


def calibrate(campaign, state, args, device):
    """Short run of the same config on this host; abort if the full run won't fit."""
    sys.path.insert(0, str(REPO))
    from src.config import load_config
    full_bio = planned_bio_s(load_config(REPO / args.config))
    log(campaign, f"calibrating: {CALIBRATION_PRESENTATIONS} presentations "
                  f"(full run = {full_bio:.0f} s biological)")
    remaining = state["deadline_epoch"] - time.time() - COLLECT_MARGIN_S
    out = ssh(state, f"cd {REMOTE}/project && {REMOTE}/miniforge/bin/conda run -n digital-brain "
                     f"--no-capture-output python -m src.main --config {shlex.quote(args.config)}"
                     f"{device} --n-presentations {CALIBRATION_PRESENTATIONS} --no-plots "
                     f"--out {REMOTE}/calibration > {REMOTE}/calibration.log 2>&1 && "
                     f"cat {REMOTE}/calibration/*/metrics.json",
              timeout=max(60, int(remaining)))
    forecast, rtf = forecast_run_s(json.loads(out.stdout), full_bio)
    remaining = state["deadline_epoch"] - time.time() - COLLECT_MARGIN_S
    state.update(calibration_rtf=rtf, forecast_s=forecast)
    save_state(campaign, state)
    log(campaign, f"calibration RTF {rtf:.2f}: full run forecast {forecast / 60:.0f} min "
                  f"(x{FORECAST_SAFETY} margin), {remaining / 60:.0f} min left before collection")
    if forecast > remaining:
        raise RuntimeError(f"forecast {forecast / 60:.0f} min exceeds the {remaining / 60:.0f} min "
                           "left under the cap; aborting before the full run")


def wait_for_ssh(campaign, state):
    deadline = min(time.time() + 1200, state["deadline_epoch"] - COLLECT_MARGIN_S)
    last = None
    while time.time() < deadline:
        info = api().show_instance(state["instance_id"]) or {}
        status = (info.get("actual_status"), info.get("status_msg"))
        if status != last:  # so a host that never comes up can be diagnosed
            log(campaign, f"instance status: {status[0]} | {(status[1] or '').strip()[:160]}")
            last = status
        if info.get("actual_status") == "running" and info.get("ssh_host"):
            state.update(ssh_host=info["ssh_host"], ssh_port=info["ssh_port"])
            if ssh(state, "true", timeout=60, check=False).returncode == 0:
                save_state(campaign, state)
                log(campaign, f"ssh ready at {state['ssh_host']}:{state['ssh_port']}")
                return
        time.sleep(15)
    raise RuntimeError("instance never became reachable over ssh")


def collect(campaign, state):
    """Bring back logs first, then run outputs; small files before big ones."""
    ssh(state, f"nvidia-smi -q > {REMOTE}/nvidia-smi.txt 2>&1; cd {REMOTE} && "
               f"tar -czf results.tar.gz --ignore-failed-read --exclude=build setup.log "
               f"calibration.log run.log run.exit nvidia-smi.txt calibration project/data/runs "
               f"2>/dev/null; true", check=False)
    try:
        scp(state, f"root@{state['ssh_host']}:{REMOTE}/results.tar.gz", str(campaign))
        subprocess.run(["tar", "-xzf", str(campaign / "results.tar.gz"), "-C", str(campaign)],
                       check=True)
        log(campaign, f"results collected into {campaign}")
    except Exception as exc:  # noqa: BLE001
        log(campaign, f"collection failed: {exc}")


def cmd_cleanup(args):
    campaign = Path(args.campaign)
    state = load_state(campaign)
    record = destroy_owned(state)
    state.update(status="destroyed" if record["verified_absent"] else "destroy_unverified",
                 termination=record | {"by": "cleanup"})
    save_state(campaign, state)
    print(json.dumps(record, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quote", help="read-only offer search and lifetime caps")
    q.add_argument("--gpu", default="H100_SXM")
    q.add_argument("--max-cost", type=float, default=3.0)
    q.add_argument("--top", type=int, default=5)
    q.add_argument("--country", default=None, help="two-letter code, e.g. US")
    la = sub.add_parser("launch", help="rent, run one config, collect, destroy")
    la.add_argument("--machine-id", type=int, default=None,
                    help="from `quote`; omit to take the cheapest eligible offer within --max-hourly")
    la.add_argument("--max-hourly", type=float, required=True,
                    help="approved USD/h including disk; also sets the lifetime cap")
    la.add_argument("--gpu", default="H100_SXM")
    la.add_argument("--config", required=True)
    la.add_argument("--device", default=None)
    la.add_argument("--max-cost", type=float, required=True, help="hard USD cap for this rental")
    la.add_argument("--max-hours", type=float, default=6.0)
    la.add_argument("--country", default=None, help="two-letter code, e.g. US")
    la.add_argument("--exclude-machines", type=lambda v: {int(x) for x in v.split(",")},
                    default=set(), help="comma-separated machine ids to skip (e.g. a host that never came up)")
    la.add_argument("--yes", action="store_true", help="actually rent (owner-approved quote)")
    c = sub.add_parser("cleanup", help="destroy a campaign's instance and verify")
    c.add_argument("--campaign", required=True)
    w = sub.add_parser("_watchdog")
    w.add_argument("--campaign", required=True)
    args = p.parse_args()

    if args.cmd == "quote":
        cmd_quote(args)
    elif args.cmd == "launch":
        cmd_launch(args)
    elif args.cmd == "cleanup":
        cmd_cleanup(args)
    else:
        watchdog(Path(args.campaign))


if __name__ == "__main__":
    main()
