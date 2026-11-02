"""Interactive viewer for the eddy-current simulator.

    conda activate ll4ma
    cd src
    python -m viewer                    # then open http://localhost:8080
    python -m viewer --port 8081 --processes 6
    python -m viewer --processes 1      # no worker pool; every solve runs in this process

Pick settings in the side panel, then start a mode from its tab: Experiment (main.py's
task with Griffin's controller), Live goal (drag the gizmo; the controller chases it), or
Teleop (drive one magnet by hand). Headless checks: python -m viewer.selfcheck
"""
import argparse
import os
import threading


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="0.0.0.0", help="address to serve on")
    ap.add_argument("--port", type=int, default=8080, help="port to serve on")
    ap.add_argument(
        "--processes",
        type=int,
        default=10,
        help="solver pool size (main.py uses 10); 1 solves in this process",
    )
    args = ap.parse_args()
    # Griffin's modules import pyplot; the pool's workers inherit this too.
    os.environ.setdefault("MPLBACKEND", "Agg")

    import viser

    from . import engine
    from .app import Viewer

    pool = None
    if args.processes > 1:
        pool = engine.make_pool(args.processes)
        # Load torch and the solvers in every worker now rather than on the first solve.
        threading.Thread(
            target=pool.map, args=(engine.warm_worker, range(args.processes)), daemon=True
        ).start()
    try:
        viewer = Viewer(viser.ViserServer(host=args.host, port=args.port), pool)
        print(f"open http://localhost:{args.port}")
        viewer.loop()
    except KeyboardInterrupt:
        pass
    finally:
        if pool is not None:
            pool.terminate()


if __name__ == "__main__":
    main()
