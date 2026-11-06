"""All local workflow stages, with explicit inputs and fresh output paths."""
import argparse
import json
from .io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Bicyclic peptide design workflow for PSMA and FAP")
    commands = parser.add_subparsers(dest="command", required=True)
    target = commands.add_parser("prepare-target")
    target.add_argument("--pdb", required=True)
    target.add_argument("--config", required=True)
    target.add_argument("--out", required=True)
    triangle = commands.add_parser("screen")
    triangle.add_argument("--pdb", required=True)
    triangle.add_argument("--chain", required=True)
    triangle.add_argument("--sequence")
    triangle.add_argument("--mode", choices=["fixed_ends","flexible"], default="flexible")
    triangle.add_argument("--min-length", type=int, default=13)
    triangle.add_argument("--allow-four-cys", action="store_true")
    triangle.add_argument("--out", required=True)
    rfd = commands.add_parser("plan-rfdiffusion")
    rfd.add_argument("--config", required=True)
    rfd.add_argument("--profile", required=True)
    rfd.add_argument("--design-dir", required=True)
    rfd.add_argument("--out", required=True)
    mpnn = commands.add_parser("prepare-mpnn")
    mpnn.add_argument("--pdb", required=True)
    mpnn.add_argument("--chain", required=True)
    mpnn.add_argument("--profile", required=True)
    mpnn.add_argument("--out", required=True)
    mpnn.add_argument("--mode", choices=["fixed_ends","flexible"], default="flexible")
    mpnn.add_argument("--min-length", type=int, default=13)
    mpnn.add_argument("--num-sequences", type=int, default=10)
    mpnn.add_argument("--temperature", type=float, default=.1)
    mpnn.add_argument("--seed", type=int, default=1)
    cp = commands.add_parser("prepare-composer")
    cp.add_argument("--pdb", required=True)
    cp.add_argument("--receptor-chain", required=True)
    cp.add_argument("--peptide-chain", required=True)
    cp.add_argument("--profile", required=True)
    cp.add_argument("--num-samples", type=int, default=5)
    cp.add_argument("--out", required=True)
    mpnn_screen = commands.add_parser("screen-mpnn")
    mpnn_screen.add_argument("--fasta", required=True)
    mpnn_screen.add_argument("--pdb", required=True)
    mpnn_screen.add_argument("--chain", required=True)
    mpnn_screen.add_argument("--prefix", required=True)
    mpnn_screen.add_argument("--min-length", type=int, default=13)
    mpnn_screen.add_argument("--out", required=True)
    cp_screen = commands.add_parser("screen-composer")
    cp_screen.add_argument("--jsonl", required=True)
    cp_screen.add_argument("--min-length", type=int, default=13)
    cp_screen.add_argument("--allow-four-cys", action="store_true")
    cp_screen.add_argument("--out", required=True)
    task = commands.add_parser("build-hf3-task")
    task.add_argument("--spec", required=True)
    task.add_argument("--out", required=True)
    batch = commands.add_parser("build-task-batch")
    batch.add_argument("--candidates", required=True)
    batch.add_argument("--target", required=True)
    batch.add_argument("--linker", required=True)
    batch.add_argument("--out", required=True)
    batch.add_argument("--head-to-tail", action="store_true")
    batch.add_argument("--allow-extra-cys", action="store_true")
    prediction = commands.add_parser("plan-prediction")
    prediction.add_argument("--manifest", required=True)
    prediction.add_argument("--backend", required=True)
    prediction.add_argument("--out", required=True)
    interface = commands.add_parser("interface")
    interface.add_argument("--cif", required=True)
    interface.add_argument("--peptide-chain", required=True)
    interface.add_argument("--target-chain", required=True)
    interface.add_argument("--numbering", choices=["auth","label"], required=True)
    interface.add_argument("--cutoff", type=float, default=8.)
    interface.add_argument("--out", required=True)
    collection = commands.add_parser("collect")
    collection.add_argument("--manifest", required=True)
    collection.add_argument("--mapping", required=True)
    collection.add_argument("--out", required=True)
    rank = commands.add_parser("rank")
    rank.add_argument("--csv", required=True)
    rank.add_argument("--config", required=True)
    rank.add_argument("--out", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--gpu")
    args = parser.parse_args()
    try:
        if args.command == "prepare-target":
            from .structure import prepare_target
            cfg = read_json(args.config)
            result = str(prepare_target(args.pdb,args.out,cfg["chain"],cfg["ranges"],cfg["hotspots"]))
        elif args.command == "screen":
            from .structure import pdb_residues
            from .geometry import triangle_search, screen_sequence
            coords = [r["atoms"]["CA"] for r in pdb_residues(args.pdb,args.chain)]
            options = {"mode":args.mode,"min_length":args.min_length}
            result = (screen_sequence(args.sequence, coords, allowed_cys_counts=(3,4) if args.allow_four_cys else (3,), **options)
                      if args.sequence else triangle_search(coords, **options))
            write_json(args.out, result)
        elif args.command == "plan-rfdiffusion":
            from .engines import rfdiffusion_plan
            result = rfdiffusion_plan(read_json(args.config),read_json(args.profile),args.design_dir)
            write_json(args.out,result)
        elif args.command == "prepare-mpnn":
            from .engines import prepare_mpnn
            result = prepare_mpnn(args.pdb,args.chain,args.out,read_json(args.profile),args.mode,
                                  args.min_length,args.num_sequences,args.temperature,args.seed)
        elif args.command == "prepare-composer":
            from .composer import prepare
            result = prepare(args.pdb,args.receptor_chain,args.peptide_chain,args.out,read_json(args.profile),args.num_samples)
        elif args.command == "build-hf3-task":
            from .hf3 import build_task
            result = build_task(**read_json(args.spec))
            write_json(args.out,result)
        elif args.command == "screen-mpnn":
            from .sequences import mpnn_records, screen_records
            result = screen_records(mpnn_records(args.fasta,args.pdb,args.chain,args.prefix),args.out,args.min_length)
        elif args.command == "build-task-batch":
            from .prediction import build_batch
            result = build_batch(read_json(args.candidates),read_json(args.target),read_json(args.linker),
                                 args.out,args.head_to_tail,args.allow_extra_cys)
        elif args.command == "plan-prediction":
            from .prediction import plan_batch
            result = plan_batch(read_json(args.manifest),read_json(args.backend),args.out)
        elif args.command == "screen-composer":
            from .sequences import composer_records, screen_records
            result = screen_records(composer_records(args.jsonl),args.out,args.min_length,args.allow_four_cys)
        elif args.command == "interface":
            from .structure import cif_interface
            result = cif_interface(args.cif,args.peptide_chain,args.target_chain,args.cutoff,args.numbering)
            write_json(args.out,result)
        elif args.command == "collect":
            from .collect import collect
            result = collect(read_json(args.manifest),read_json(args.mapping),args.out)
        elif args.command == "rank":
            from .ranking import rank_csv
            result = rank_csv(args.csv,args.out,read_json(args.config))
        elif args.command == "run":
            from .engines import execute
            result = execute(read_json(args.plan),args.out,args.gpu)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
