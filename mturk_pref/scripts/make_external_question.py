"""
Generate MTurk ExternalQuestion XML pointing to a deployed HTTPS URL.

Usage:
  python3 mturk_pref/scripts/make_external_question.py --url "https://YOUR-SERVICE.onrender.com/" --height 900
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument(
        "--xmlns",
        default="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd",
    )
    args = ap.parse_args()

    url = args.url.strip()
    if not url.startswith("https://"):
        raise SystemExit("ExternalURL must be https:// for MTurk ExternalQuestion.")
    if not url.endswith("/"):
        url += "/"

    xml = (
        f'<ExternalQuestion xmlns="{args.xmlns}">\n'
        f"  <ExternalURL>{url}</ExternalURL>\n"
        f"  <FrameHeight>{int(args.height)}</FrameHeight>\n"
        f"</ExternalQuestion>\n"
    )
    sys.stdout.write(xml)


if __name__ == "__main__":
    main()
