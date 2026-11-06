import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from baidu_bicycle.geometry import triangle_search, screen_sequence
from baidu_bicycle.hf3 import build_task
from baidu_bicycle.io import helm_sequence, write_json
from baidu_bicycle.ranking import quantile, select_rows, rank_csv
from baidu_bicycle.engines import execute, rfdiffusion_plan, prepare_mpnn
from baidu_bicycle.structure import pdb_residues, prepare_target
from baidu_bicycle.composer import prepare as prepare_composer


def coords():
    points = [[30.+i,30.,30.] for i in range(13)]
    points[0] = [0.,0.,0.]
    points[6] = [8.,0.,0.]
    points[12] = [4.,math.sqrt(48),0.]
    return points


def pdb(path):
    lines = []
    serial = 1
    for chain, points in (("A", coords()),("B", [[50.,50.,50.],[54.,50.,50.]])):
        for i,xyz in enumerate(points,1):
            for atom,offset in (("N",-.5),("CA",0.),("C",.5),("O",1.)):
                x,y,z=xyz
                lines.append(f"ATOM  {serial:5d} {atom:^4s} ALA {chain}{i:4d}    {x+offset:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           {atom[0]:>2s}")
                serial += 1
    path.write_text("\n".join(lines)+"\nEND\n")


class GeometryTests(unittest.TestCase):
    def test_equilateral_triangle(self):
        result = triangle_search(coords(),mode="fixed_ends")
        self.assertEqual(result["positions_0based"],[0,6,12])
        for d in result["distances_angstrom"]:
            self.assertAlmostEqual(d,8.)

    def test_strict_bounds(self):
        self.assertFalse(triangle_search([[0,0,0],[6,0,0],[3,math.sqrt(27),0]],spacing=1,min_length=3)["accepted"])

    def test_short_backbone(self):
        self.assertEqual(triangle_search(coords()[:10])["reason"],"peptide_too_short")

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            triangle_search([[float("nan"),0,0]]*13)

    def test_no_spurious_success(self):
        self.assertFalse(triangle_search([[i*100.,0,0] for i in range(13)])["accepted"])

    def test_exact_three_cysteines(self):
        self.assertTrue(screen_sequence("CAAAAACAAAAAC",coords())["accepted"])
        self.assertEqual(screen_sequence("CCAAAACAAAAAC",coords())["reason"],"cysteine_count")

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            screen_sequence("CCC",coords())

    def test_helm_tokens(self):
        self.assertEqual(helm_sequence("PEPTIDE1{C.A.C.A.C}|CHEM1{LFI}$$$$"),"CACAC")
        with self.assertRaises(ValueError):
            helm_sequence("PEPTIDE1{C.[meA].C}$$$$")


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.config = {"interface_column":"contacts","mapping_verified":True,"hotspots":["101"],
                       "plddt_quantile":.5,"iptm_quantile":.5,"hotspot_fraction":.2,
                       "residue_map":{"1":"101","2":"102"}}
        self.rows = [{"task_name":str(i),"peptide_chain_plddt":p,"global_iptm":s,
                      "contacts":"1,1,2","interface_threshold_angstrom":8} for i,p,s in [(0,60,.2),(1,80,.9),(2,90,.8)]]

    def test_linear_quantile_and_intersection(self):
        self.assertAlmostEqual(quantile([1,2,3,4],.98),3.94)
        rows, stats=select_rows(self.rows,self.config)
        self.assertEqual(stats["selected_rows"],2)
        self.assertEqual(rows[1]["hotspot_fraction_recomputed"],.5)

    def test_missing_interface_fails(self):
        del self.rows[0]["contacts"]
        with self.assertRaises(ValueError): select_rows(self.rows,self.config)

    def test_unverified_mapping_fails(self):
        self.config["mapping_verified"]=False
        with self.assertRaises(ValueError): select_rows(self.rows,self.config)

    def test_unmapped_contact_fails(self):
        self.rows[0]["contacts"]="999"
        with self.assertRaises(ValueError): select_rows(self.rows,self.config)

    def test_duplicate_task_fails(self):
        self.rows.append(self.rows[0])
        with self.assertRaises(ValueError): select_rows(self.rows,self.config)

    def test_nan_fails(self):
        self.rows[0]["global_iptm"]="nan"
        with self.assertRaises(ValueError): select_rows(self.rows,self.config)

    def test_wrong_threshold_fails(self):
        self.rows[0]["interface_threshold_angstrom"]=10
        with self.assertRaises(ValueError): select_rows(self.rows,self.config)


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.linker={"ccd":"LFI","reactive_atoms":["C11","C10","C12"],"chemistry_verified":True}

    def test_covalent_positions_and_topology(self):
        task=build_task("synthetic","CAAAAACAAAAAC","AAAA",self.linker,[1,7,13])
        self.assertEqual([b["left_position"] for b in task["covalent_bonds"]],[1,7,13])
        self.assertEqual(len(task["covalent_bonds"]),3)
        task=build_task("synthetic","CAAAAACAAAAAC","AAAA",self.linker,[1,7,13],head_to_tail=True)
        self.assertEqual(len(task["covalent_bonds"]),4)

    def test_unknown_chemistry(self):
        self.linker["chemistry_verified"]=False
        with self.assertRaises(ValueError): build_task("x","CACAC","A",self.linker,[1,3,5])

    def test_bad_positions(self):
        with self.assertRaises(ValueError): build_task("x","CACAC","A",self.linker,[1,2,5])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)
        self.pdb=self.path/"input.pdb"
        pdb(self.pdb)

    def tearDown(self):
        self.tmp.cleanup()

    def test_target_mapping(self):
        prepare_target(self.pdb,self.path/"target","B",[[1,2]],[2])
        d=json.loads((self.path/"target/target.json").read_text())
        self.assertEqual(d["hotspots_entity_1based"],[2])
        self.assertEqual(d["sequence"],"AA")

    def test_unresolved_target_residue(self):
        with self.assertRaises(ValueError): prepare_target(self.pdb,self.path/"target","B",[[1,3]],[2])

    def test_mpnn_fixed_residues_are_mutated(self):
        profile={"proteinmpnn_root":str(self.path),"proteinmpnn_python":sys.executable,
                 "proteinmpnn_weights":str(self.path)}
        plan=prepare_mpnn(self.pdb,"A",self.path/"mpnn",profile,mode="fixed_ends")
        residues=pdb_residues(self.path/"mpnn/backbone.pdb","A")
        self.assertEqual("".join(r["aa"] for r in residues),"CAAAAACAAAAAC")
        fixed=json.loads((self.path/"mpnn/fixed_positions.jsonl").read_text())
        self.assertEqual(fixed,{"backbone":{"A":[1,7,13],"B":[]}})
        self.assertEqual(len((self.path/"mpnn/fixed_positions.jsonl").read_text().splitlines()),1)
        self.assertIn("CX",plan["argv"])

    def test_composer_plan(self):
        profile={"composer_root":str(self.path),"composer_python":sys.executable,"composer_checkpoint":"weights.ckpt"}
        plans=prepare_composer(self.pdb,"B","A",self.path/"composer",profile)
        cfg=json.loads((self.path/"composer/generation.json").read_text())
        self.assertFalse(cfg["CADS_sampler"])
        self.assertEqual(plans["generation"]["env"]["condition"],"4")

    def test_cpu_runner_records_failure(self):
        plan={"argv":[sys.executable,"-c","raise SystemExit(7)"],"cwd":str(self.path),"inputs":[str(self.pdb)]}
        with self.assertRaises(RuntimeError): execute(plan,self.path/"run")
        record=json.loads((self.path/"run/finished.json").read_text())
        self.assertEqual(record["returncode"],7)
        self.assertEqual(record["status"],"failed")

    def test_cpu_runner_success_and_no_overwrite(self):
        plan={"argv":[sys.executable,"-c","print('fixture')"],"cwd":str(self.path)}
        self.assertEqual(execute(plan,self.path/"run")["status"],"complete")
        with self.assertRaises(FileExistsError): execute(plan,self.path/"run")

    def test_rfdiffusion_plan_preserves_interpreter(self):
        cfg={"peptide_length":13,"triangle_positions_0based":[0,6,12],"guide_scale":2.,"k_dist":5.,
             "potential_version":"bicycle_triangle_v1","target_pdb":str(self.pdb),"num_designs":1,
             "target_contig":"B1-2","hotspots":["B2"]}
        plan=rfdiffusion_plan(cfg,{"rfdiffusion_root":str(self.path),"rfdiffusion_python":sys.executable,
                                  "rfdiffusion_checkpoint":"model.pt"},self.path/"design")
        self.assertEqual(plan["argv"][0],str(Path(sys.executable).absolute()))
        self.assertIn("contigmap.contigs=[13-13 B1-2/0]",plan["argv"])

    def test_mpnn_native_row_excluded_and_samples_screened(self):
        from baidu_bicycle.sequences import mpnn_records, screen_records
        fasta=self.path/'samples.fa'
        fasta.write_text('>native, score=1\nAAAAAAAAAAAAA\n>T=0.1, sample=1, score=0.3\nCAAAAACAAAAAC\n')
        rows=mpnn_records(fasta,self.pdb,'A','fixture')
        self.assertEqual(len(rows),1)
        result=screen_records(rows,self.path/'screen')
        self.assertEqual(result,{'total':1,'accepted':1})

    def test_collect_requires_same_structure(self):
        from baidu_bicycle.collect import collect
        from baidu_bicycle.io import sha256
        write_json(self.path/'metrics.json',{'iptm':.9,'peptide':80.})
        interface={'target_interface_residues':['2'],'numbering':'auth','target_chain':'B',
                   'structure_sha256':sha256(self.pdb),'interface_threshold_angstrom':8}
        write_json(self.path/'interface.json',interface)
        manifest=[{'task_name':'fixture','metrics_json':str(self.path/'metrics.json'),
                   'interface_json':str(self.path/'interface.json'),'peptide_helm':'PEPTIDE1{A}$$$$',
                   'cif_file':str(self.pdb)}]
        result=collect(manifest,{'global_iptm':['iptm'],'peptide_chain_plddt':['peptide']},self.path/'collected')
        self.assertEqual(result['collected'],1)
        self.pdb.write_text('changed')
        with self.assertRaises(ValueError):
            collect(manifest,{'global_iptm':['iptm'],'peptide_chain_plddt':['peptide']},self.path/'wrong')

    def test_optional_cif_contact_numbering(self):
        try:
            from Bio.PDB import PDBParser, MMCIFIO
        except ImportError:
            self.skipTest('Biopython optional dependency')
        from baidu_bicycle.structure import cif_interface
        model=PDBParser(QUIET=True).get_structure('fixture',str(self.pdb))
        # Move one target residue onto the peptide triangle; the other remains distant.
        for atom in model[0]['B'][1]: atom.coord[:]=[0.,0.,0.]
        writer=MMCIFIO(); writer.set_structure(model); writer.save(str(self.path/'fixture.cif'))
        result=cif_interface(self.path/'fixture.cif','A','B',cutoff=8.,numbering='auth')
        self.assertEqual(result['target_interface_residues'],['1'])


if __name__ == "__main__": unittest.main()
