#!/usr/bin/env python3
"""
バージョン更新スクリプト

使用方法:
    python scripts/update_version.py 0.2.0
    python scripts/update_version.py --patch  # 0.1.0 -> 0.1.1
    python scripts/update_version.py --minor  # 0.1.0 -> 0.2.0
    python scripts/update_version.py --major  # 0.1.0 -> 1.0.0
"""

import re
import sys
import argparse
from pathlib import Path


def read_version():
    """pyproject.tomlから現在のバージョンを読み取る"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    match = re.search(r'^version = ["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise ValueError("Could not find version in pyproject.toml")


def write_version(version):
    """pyproject.tomlにバージョンを書き込む"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = re.sub(
        r'^version = ["\'][^"\']+["\']',
        f'version = "{version}"',
        content,
        flags=re.MULTILINE
    )
    
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✓ Updated version to {version} in pyproject.toml")


def increment_version(version, part):
    """バージョンをインクリメント"""
    parts = version.split("-")
    base_version = parts[0]
    prerelease = "-".join(parts[1:]) if len(parts) > 1 else None
    
    major, minor, patch = map(int, base_version.split("."))
    
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid part: {part}")
    
    new_version = f"{major}.{minor}.{patch}"
    if prerelease:
        new_version = f"{new_version}-{prerelease}"
    
    return new_version


def main():
    parser = argparse.ArgumentParser(description="Update package version")
    parser.add_argument(
        "version",
        nargs="?",
        help="New version (e.g., 0.2.0) or use --major, --minor, --patch"
    )
    parser.add_argument("--major", action="store_true", help="Increment major version")
    parser.add_argument("--minor", action="store_true", help="Increment minor version")
    parser.add_argument("--patch", action="store_true", help="Increment patch version")
    
    args = parser.parse_args()
    
    current_version = read_version()
    print(f"Current version: {current_version}")
    
    if args.version:
        new_version = args.version
    elif args.major:
        new_version = increment_version(current_version, "major")
    elif args.minor:
        new_version = increment_version(current_version, "minor")
    elif args.patch:
        new_version = increment_version(current_version, "patch")
    else:
        parser.print_help()
        sys.exit(1)
    
    # バージョン形式の検証
    if not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$', new_version):
        print(f"ERROR: Invalid version format: {new_version}")
        print("Version must follow Semantic Versioning: MAJOR.MINOR.PATCH[-prerelease]")
        sys.exit(1)
    
    write_version(new_version)
    print(f"\nNext steps:")
    print(f"  1. git add pyproject.toml")
    print(f"  2. git commit -m 'Bump version to {new_version}'")
    print(f"  3. git tag v{new_version}")
    print(f"  4. git push origin main --tags")


if __name__ == "__main__":
    main()

