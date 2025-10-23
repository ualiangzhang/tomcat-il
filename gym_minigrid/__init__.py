from pathlib import Path
import sys
import warnings

# Add `gym_minigrid` parent directory to system path
path = str(Path(__file__).parent.parent.resolve().absolute())
sys.path.insert(0, path)

# Import the envs module so that envs register themselves
import gym_minigrid.envs

# Import wrappers if available; some environments (e.g., visualization-only)
# do not require wrappers and certain gym versions may not expose GoalEnv.
try:
    import gym_minigrid.wrappers  # type: ignore
except Exception as e:  # pragma: no cover
    warnings.warn(f"gym_minigrid.wrappers import skipped: {e}")
