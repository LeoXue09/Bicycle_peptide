#!/usr/bin/env python3
"""Synthetic, offline end-to-end demonstration of custom preprocessing/postprocessing."""
import argparse
import csv
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baidu_bicycle.geometry import screen_sequence
from baidu_bicycle.hf3 import build_task
from baidu_bicycle.io import write_json
from baidu_bicycle.ranking import rank_csv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    xyz=[[30.+i,30.,30.] for i in range(13)]
    xyz[0],xyz[6],xyz[12]=[0.,0.,0.],[8.,0.,0.],[4.,math.sqrt(48),0.]
    peptide="CAAAAACAAAAAC"
    result=screen_sequence(peptide,xyz,mode="fixed_ends")
    assert result["accepted"]
    write_json(args.out/"triangle.json",result)
    # chemistry_verified here identifies a schema fixture, not molecular validation.
    task=build_task("synthetic_schema_fixture",peptide,"AAAA",
                    {"ccd":"LFI","reactive_atoms":["C11","C10","C12"],"chemistry_verified":True},[1,7,13])
    write_json(args.out/"task.json",task)
    with (args.out/"synthetic_metrics.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["task_name","peptide_chain_plddt","global_iptm","contacts","interface_threshold_angstrom"])
        writer.writerows([["synthetic_low",50,.2,"1,2",8],["synthetic_middle",80,.7,"1,2",8],["synthetic_high",90,.9,"1,2",8]])
    config={"interface_column":"contacts","mapping_verified":True,"hotspots":["1"],
            "plddt_quantile":.98,"iptm_quantile":.98,"hotspot_fraction":.2,
            "population":"synthetic software fixture, not a scientific result"}
    stats=rank_csv(args.out/"synthetic_metrics.csv",args.out/"selection",config)
    assert stats["selected_rows"]==1
    write_json(args.out/"README.json",{"synthetic":True,"model_inference":False,"expected_selected_task":"synthetic_high"})
    print(f"Synthetic demo passed: {args.out}")


if __name__ == "__main__": main()
