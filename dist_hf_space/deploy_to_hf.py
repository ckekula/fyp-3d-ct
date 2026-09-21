#!/usr/bin/env python3
"""
deploy_to_hf.py
Helper script to deploy ThorAxis to Hugging Face Spaces in 1 command.

Usage:
  python3 eval/webapp/deploy_to_hf.py --space-id LCKSVD/ThorAxis
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Deploy ThorAxis to Hugging Face Spaces.")
    parser.add_argument("--space-id", required=True, help="Your Hugging Face Space ID (e.g. LCKSVD/ThorAxis)")
    parser.add_argument("--token", help="Optional HF write token")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    webapp_dir = repo_root / "eval" / "webapp"
    eval_core_dir = repo_root / "eval" / "core"
    eval_adapters_dir = repo_root / "eval" / "adapters"
    leaderboard_dir = repo_root / "leaderboard"
    
    print("=" * 70)
    print(f"🚀 Staging ThorAxis Leaderboard for Hugging Face Space: {args.space_id}")
    print("=" * 70)
    
    # 1. Prepare export directory
    export_dir = repo_root / "dist_hf_space"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    ignore_func = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git*", ".pytest_cache", "tests")
    
    print(f"📦 Copying webapp files to staging directory: {export_dir}")
    shutil.copytree(webapp_dir, export_dir / "app_src", dirs_exist_ok=True, ignore=ignore_func)
    
    # Copy eval.core and eval.adapters
    shutil.copytree(eval_core_dir, export_dir / "eval" / "core", dirs_exist_ok=True, ignore=ignore_func)
    shutil.copytree(eval_adapters_dir, export_dir / "eval" / "adapters", dirs_exist_ok=True, ignore=ignore_func)
    
    # Copy leaderboard module (validation, schemas, examples, docs)
    shutil.copytree(leaderboard_dir, export_dir / "leaderboard", dirs_exist_ok=True, ignore=ignore_func)
    
    # Move app files to root of export
    for item in (export_dir / "app_src").iterdir():
        dest = export_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True, ignore=ignore_func)
        else:
            shutil.copy2(item, dest)
    shutil.rmtree(export_dir / "app_src")

    # Add clean .gitignore
    gitignore_path = export_dir / ".gitignore"
    gitignore_path.write_text("__pycache__/\n*.py[cod]\n*$py.class\n.env\n.DS_Store\n")
    
    print("\n✅ Staging directory prepared successfully!")
    print("\nTo push to your Hugging Face Space, run the following commands:")
    print("-" * 70)
    print(f"cd {export_dir}")
    print("git init")
    print("git branch -M main")
    print(f"git remote add origin https://huggingface.co/spaces/{args.space_id}")
    print("git add .")
    print("git commit -m 'Release ThorAxis Leaderboard & Benchmark'")
    print("git push --set-upstream origin main --force")
    print("-" * 70)
    print(f"🎉 Once pushed, your app will be live at: https://huggingface.co/spaces/{args.space_id}\n")

if __name__ == "__main__":
    main()
