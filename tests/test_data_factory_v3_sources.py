import json
import tempfile
import unittest
from pathlib import Path

from sieve.data_factory.v3_sources import load_car_tasks, load_tau3_tasks


class V3SourceAdapterTests(unittest.TestCase):
    def test_tau3_adapter_keeps_train_and_rejects_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "data" / "tau2" / "domains" / "airline"
            domain.mkdir(parents=True)
            (domain / "split_tasks.json").write_text(
                json.dumps({"train": ["0"], "test": ["1"]}), encoding="utf-8"
            )
            tasks = [
                {
                    "id": 0,
                    "user_scenario": {"instructions": {"domain": "airline", "reason_for_call": "Change seat"}},
                    "evaluation_criteria": {"actions": [{"name": "update_seat", "arguments": {"reservation_id": "R1", "seat": "12A"}}]},
                },
                {
                    "id": 1,
                    "user_scenario": {"instructions": {"domain": "airline", "reason_for_call": "Held out"}},
                    "evaluation_criteria": {"actions": [{"name": "update_seat", "arguments": {"reservation_id": "R2", "seat": "14B"}}]},
                },
            ]
            (domain / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")

            loaded = load_tau3_tasks(root, domains=("airline",))

            self.assertEqual([task.parent_task_id for task in loaded], ["tau3:airline:0"])
            self.assertEqual(loaded[0].reference_actions[0]["name"], "update_seat")
            self.assertEqual(loaded[0].source_split, "train")

    def test_car_adapter_parses_reference_python_without_importing_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks_dir = root / "docs" / "reference_data" / "tasks"
            tasks_dir.mkdir(parents=True)
            (tasks_dir / "task_splits.json").write_text(
                json.dumps({"base_train": ["base_0"], "base_test": ["base_1"], "disambiguation_train": [], "disambiguation_test": []}),
                encoding="utf-8",
            )
            (tasks_dir / "tasks_base.py").write_text(
                "TASKS=[Task(task_id='base_0',instruction='Open the trunk',context_init_config={'trunk':'CLOSED'},actions=[Action(name='open_close_trunk_door',kwargs={'action':'OPEN'},index=0)]),Task(task_id='base_1',instruction='Held out',context_init_config={},actions=[])]",
                encoding="utf-8",
            )
            (tasks_dir / "tasks_disambiguation.py").write_text("TASKS=[]", encoding="utf-8")

            loaded = load_car_tasks(root)

            self.assertEqual([task.parent_task_id for task in loaded], ["car:base_0"])
            self.assertEqual(loaded[0].reference_actions[0]["name"], "open_close_trunk_door")
            self.assertEqual(loaded[0].initial_state["trunk"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
