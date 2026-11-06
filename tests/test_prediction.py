import json
from pathlib import Path
import sys
import tempfile
import unittest

from baidu_bicycle.engines import execute
from baidu_bicycle.io import sha256
from baidu_bicycle.prediction import build_batch, plan_batch, SCHEMA


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.structure = self.root / "fixture.pdb"
        self.structure.write_text("synthetic structure identity fixture\n")
        self.records = [{"task_name":"fixture_0", "pdb_path":str(self.structure),
                         "sequence":"CAAAAACAAAAAC", "structure_sha256":sha256(self.structure),
                         "screen":{"accepted":True,"positions_0based":[0,6,12]}}]
        self.linker = {"ccd":"LFI","reactive_atoms":["C11","C10","C12"],"chemistry_verified":True}
        self.backend = {"input_schema":SCHEMA,"argv":[sys.executable,"adapter.py","{input_json}","{output_dir}"],
                        "cwd":str(self.root),"gpu_required":False,"expected_files":["metrics.json"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_preserves_covalent_bonds(self):
        manifest = build_batch(self.records,{"sequence":"AAAA"},self.linker,self.root/"tasks")
        task = json.loads(Path(manifest["jobs"][0]["input_json"]).read_text())
        self.assertEqual([b["left_position"] for b in task["covalent_bonds"]],[1,7,13])
        self.assertEqual(manifest["schema"],SCHEMA)

    def test_changed_structure_rejected(self):
        self.structure.write_text("modified")
        with self.assertRaises(ValueError):
            build_batch(self.records,{"sequence":"AAAA"},self.linker,self.root/"tasks")

    def test_backend_contract_rejected(self):
        manifest = build_batch(self.records,{"sequence":"AAAA"},self.linker,self.root/"tasks")
        self.backend["input_schema"]="sequence-only"
        with self.assertRaises(ValueError): plan_batch(manifest,self.backend,self.root/"plans")

    def test_output_traversal_rejected(self):
        manifest = build_batch(self.records,{"sequence":"AAAA"},self.linker,self.root/"tasks")
        self.backend["expected_files"]=["../outside.json"]
        with self.assertRaises(ValueError): plan_batch(manifest,self.backend,self.root/"plans")

    def test_task_hash_verified(self):
        manifest = build_batch(self.records,{"sequence":"AAAA"},self.linker,self.root/"tasks")
        Path(manifest["jobs"][0]["input_json"]).write_text("{}")
        with self.assertRaises(ValueError): plan_batch(manifest,self.backend,self.root/"plans")

    def test_mock_backend_execution(self):
        # This fixture tests orchestration only; it performs no structure prediction.
        (self.root/"adapter.py").write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "task=json.loads(Path(sys.argv[1]).read_text())\n"
            "assert len(task['covalent_bonds'])==3\n"
            "out=Path(sys.argv[2]);out.mkdir(parents=True)\n"
            "(out/'metrics.json').write_text(json.dumps({'synthetic':True,'bonds':3}))\n")
        manifest=build_batch(self.records,{"sequence":"AAAA"},self.linker,self.root/"tasks")
        plans=plan_batch(manifest,self.backend,self.root/"plans")
        plan=json.loads(Path(plans["plans"][0]).read_text())
        receipt=execute(plan,self.root/"execution")
        self.assertEqual(receipt["status"],"complete")
        self.assertEqual(json.loads(Path(plan["expected_outputs"][0]).read_text())["bonds"],3)


if __name__ == "__main__": unittest.main()
