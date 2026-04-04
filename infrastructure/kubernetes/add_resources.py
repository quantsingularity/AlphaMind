#!/usr/bin/env python3
"""Add resource limits and health probes to Kubernetes Deployment manifests."""

import sys
from pathlib import Path

import yaml


def add_resources_to_deployment(file_path: Path) -> bool:
    """Add resource limits and probes to a Deployment if missing."""
    try:
        with open(file_path, "r") as f:
            docs = list(yaml.safe_load_all(f))
    except (OSError, yaml.YAMLError) as e:
        print(f"ERROR reading {file_path}: {e}", file=sys.stderr)
        return False

    modified = False
    for doc in docs:
        if not doc or doc.get("kind") != "Deployment":
            continue
        containers = (
            doc.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        for container in containers:
            if "resources" not in container:
                container["resources"] = {
                    "requests": {"memory": "256Mi", "cpu": "250m"},
                    "limits": {"memory": "512Mi", "cpu": "500m"},
                }
                modified = True

            if "livenessProbe" not in container:
                container["livenessProbe"] = {
                    "httpGet": {"path": "/health", "port": 8080},
                    "initialDelaySeconds": 30,
                    "periodSeconds": 10,
                    "failureThreshold": 3,
                }
                modified = True

            if "readinessProbe" not in container:
                container["readinessProbe"] = {
                    "httpGet": {"path": "/health", "port": 8080},
                    "initialDelaySeconds": 10,
                    "periodSeconds": 5,
                    "failureThreshold": 3,
                }
                modified = True

    if modified:
        try:
            with open(file_path, "w") as f:
                yaml.dump_all(docs, f, default_flow_style=False, sort_keys=False)
        except OSError as e:
            print(f"ERROR writing {file_path}: {e}", file=sys.stderr)
            return False
        return True
    return False


if __name__ == "__main__":
    # Resolve base_dir relative to this script's location, not the CWD
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "base"

    if not base_dir.exists():
        print(f"ERROR: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    updated = []
    for file_path in sorted(base_dir.glob("*-deployment.yaml")):
        if add_resources_to_deployment(file_path):
            updated.append(file_path.name)
            print(f"Updated: {file_path.name}")

    if not updated:
        print("No deployments required changes.")
    else:
        print(f"\nDone. Updated {len(updated)} file(s).")
