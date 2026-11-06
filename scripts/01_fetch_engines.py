#!/usr/bin/env python3
"""Fetch pinned public source. No weights, datasets, environment installs or overwrites."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--engine",choices=["rfdiffusion","proteinmpnn","cp_composer","helixfold3"],action="append")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/"engines.lock.json").read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    for name in args.engine or lock:
        spec=lock[name]
        dest=args.out/name
        dest.mkdir(exist_ok=False)
        def git(*argv): subprocess.run(["git","-C",str(dest),*argv],check=True)
        git("init")
        git("remote","add","origin",spec["url"])
        git("config","remote.origin.promisor","true")
        git("config","remote.origin.partialclonefilter","blob:none")
        git("fetch","--depth","1","--filter=blob:none","origin",spec["commit"])
        if name=="helixfold3":
            git("sparse-checkout","set","apps/protein_folding/helixfold3","pahelix")
        else:
            git("sparse-checkout","set","--no-cone","/*","!/checkpoints/","!/assets/","!/models/weights/",
                "!/vanilla_model_weights/","!/soluble_model_weights/","!/ca_model_weights/",
                "!*.pt","!*.pth","!*.ckpt","!*.pdf","!*.ipynb")
        git("checkout","--detach","FETCH_HEAD")
        observed=subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()
        if observed!=spec["commit"]: raise RuntimeError("Commit mismatch")
        print(name,observed,dest)


if __name__=="__main__": main()
