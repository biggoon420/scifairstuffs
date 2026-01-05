"""
Creates an MTurk ExternalQuestion HIT via boto3 (no qualifications).

Usage (sandbox):
  python mturk_pref/scripts/04_create_hit.py --sandbox --external-url "https://YOUR.onrender.com" --assignments 10 --reward 0.01

Usage (production):
  python mturk_pref/scripts/04_create_hit.py --external-url "https://YOUR.onrender.com" --assignments 134 --reward 1.00
"""

from __future__ import annotations

import argparse
import boto3


def external_question_xml(external_url: str, frame_height: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
  <ExternalURL>{external_url}</ExternalURL>
  <FrameHeight>{int(frame_height)}</FrameHeight>
</ExternalQuestion>
""".strip()


def mturk_client(*, sandbox: bool, region: str):
    endpoint = (
        "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
        if sandbox
        else "https://mturk-requester.us-east-1.amazonaws.com"
    )
    return boto3.client("mturk", region_name=region, endpoint_url=endpoint)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-url", required=True)
    ap.add_argument("--title", default="Audio preference (15 quick choices)")
    ap.add_argument(
        "--description",
        default=(
            "Listen to 15 pairs of short clips and choose which speaker sounds more confident.\n\n"
            "Instructions:\n"
            "1) Use headphones if possible.\n"
            "2) Listen to both Clip A and Clip B.\n"
            "3) Click A IS BETTER or B IS BETTER.\n"
            "4) Use Skip only if a clip is broken/silent/non-speech.\n"
            "Time: ~3–6 minutes total."
        ),
    )
    ap.add_argument("--keywords", default="audio,comparison,preference,classification")
    ap.add_argument("--reward", type=str, default="1.00")
    ap.add_argument("--assignments", type=int, required=True)
    ap.add_argument("--frame-height", type=int, default=900)
    ap.add_argument("--lifetime-seconds", type=int, default=86400)
    ap.add_argument("--duration-seconds", type=int, default=1800)
    ap.add_argument("--auto-approval-seconds", type=int, default=259200)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()

    c = mturk_client(sandbox=args.sandbox, region=args.region)
    q = external_question_xml(args.external_url, args.frame_height)

    resp = c.create_hit(
        Title=args.title,
        Description=args.description,
        Keywords=args.keywords,
        Reward=str(args.reward),
        AssignmentDurationInSeconds=int(args.duration_seconds),
        LifetimeInSeconds=int(args.lifetime_seconds),
        AutoApprovalDelayInSeconds=int(args.auto_approval_seconds),
        MaxAssignments=int(args.assignments),
        Question=q,
    )

    hit = resp.get("HIT", {})
    hit_id = hit.get("HITId")
    hit_type_id = hit.get("HITTypeId")
    group_id = hit.get("HITGroupId")

    print("HITId:", hit_id)
    print("HITTypeId:", hit_type_id)
    print("HITGroupId:", group_id)

    if group_id:
        base = "https://workersandbox.mturk.com" if args.sandbox else "https://worker.mturk.com"
        print("Worker preview URL:", f"{base}/mturk/preview?groupId={group_id}")


if __name__ == "__main__":
    main()
