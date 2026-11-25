"""Build site/index.html from data/prices.parquet (fits every model at build time)."""

import time
from pathlib import Path

import pandas as pd

from gv.data import DATA_PATH
from gv.site import build_html, lab_data


def main() -> None:
    t0 = time.perf_counter()
    adj = pd.read_parquet(DATA_PATH)["adj_close"]
    data = lab_data(adj)
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(adj, data), encoding="utf-8")
    print(
        f"{out} ({out.stat().st_size / 1e6:.2f} MB): {len(data['symbols'])} series, "
        f"{3 * len(data['symbols'])} fits in {data['fit_seconds']} s, "
        f"built in {time.perf_counter() - t0:.1f} s"
    )
    print(f"dropped (under the minimum history): {data['dropped'] or 'none'}")
    print(f"fits that did not converge: {data['failed'] or 'none'}")


if __name__ == "__main__":
    main()
