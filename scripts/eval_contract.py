"""Optional frozen-episode contract shared by native evaluation callers."""

import json
import os
from pathlib import Path


def load_frozen_episodes():
    seed_path = os.environ.get("ROBOTWIN_SEED_LIST")
    if seed_path is None:
        return None
    seeds = [int(token) for token in Path(seed_path).read_text().split()]
    instructions = json.loads(Path(os.environ["ROBOTWIN_SEED_INSTRUCTIONS"]).read_text())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Frozen seed list must be nonempty and contain unique seeds")
    for seed in seeds:
        if not isinstance(instructions.get(str(seed)), str) or not instructions[str(seed)]:
            raise ValueError(f"Missing nonempty frozen instruction for seed {seed}")
    return seeds, instructions


def capture_policy_rng():
    import torch

    return {
        "cpu": torch.get_rng_state().numpy(),
        "cuda": torch.cuda.get_rng_state().cpu().numpy() if torch.cuda.is_available() else None,
    }


def use_fast_render(task_env):
    return (
        os.environ.get("ROBOTWIN_FAST_EVAL") == "1"
        and task_env.eval_video_path is None
        and not task_env.save_data
        and task_env.render_freq == 0
        and not task_env.crazy_random_light
    )


class EpisodeJournal:
    """Atomically persist completed episodes; reject reuse under another contract."""

    def __init__(self, path, identity, seeds, instructions):
        self.path = Path(path)
        self.contract = {"identity": identity, "seeds": seeds, "instructions": instructions}
        self.results = {}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved["contract"] != self.contract:
                raise ValueError(f"Episode resume contract mismatch: {self.path}")
            self.results = saved["results"]
            recorded = [int(seed) for seed in self.results]
            if recorded != seeds[:len(recorded)]:
                raise ValueError("Episode journal must contain an ordered seed prefix")
            if any(r["result"] not in {"Success", "Fail"} for r in self.results.values()):
                raise ValueError("Episode journal contains a nonterminal result")

    def record(self, seed, success, actions):
        self.results[str(seed)] = {
            "seed": seed, "result": "Success" if success else "Fail", "actions": actions,
        }
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"contract": self.contract, "results": self.results}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)
