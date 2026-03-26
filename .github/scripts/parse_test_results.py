#!/usr/bin/env python3
"""Parse JUnit XML test results and generate a Markdown failure summary."""

import glob
import os
import xml.etree.ElementTree as ET

MAX_MESSAGE_CHARS = 500
MAX_TEXT_CHARS = 2000

all_failures: list[dict[str, str]] = []

for xml_file in sorted(glob.glob("test-results*.xml")):
    suite_label = os.path.basename(xml_file).replace("test-results-", "").replace(".xml", "")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for testsuite in root.iter("testsuite"):
            for testcase in testsuite.findall("testcase"):
                classname = testcase.get("classname", "")
                name = testcase.get("name", "")
                test_id = f"{classname}::{name}" if classname else name
                for tag in ("failure", "error"):
                    node = testcase.find(tag)
                    if node is not None:
                        all_failures.append(
                            {
                                "suite": suite_label,
                                "test_id": test_id,
                                "kind": tag,
                                "message": (node.get("message") or "")[:MAX_MESSAGE_CHARS],
                                "text": (node.text or "")[:MAX_TEXT_CHARS],
                            }
                        )
    except (ET.ParseError, OSError) as exc:
        print(f"Warning: could not parse {xml_file}: {exc}")

lines: list[str] = []

if not all_failures:
    lines.append("## ✅ All Tests Passed")
    lines.append("")
else:
    lines.append(f"## ❌ Test Failures ({len(all_failures)} failure(s))")
    lines.append("")
    for f in all_failures:
        emoji = "❌" if f["kind"] == "failure" else "💥"
        lines.append(f"### {emoji} `{f['test_id']}`")
        lines.append("")
        lines.append(f"**Suite:** {f['suite']}")
        lines.append("")
        if f["message"]:
            lines.append(f"**Message:** {f['message']}")
            lines.append("")
        if f["text"]:
            lines.append("<details><summary>Details</summary>")
            lines.append("")
            lines.append("```")
            lines.append(f["text"])
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

content = "\n".join(lines)

with open("test-failures-summary.md", "w") as fh:
    fh.write(content)

print(content)
