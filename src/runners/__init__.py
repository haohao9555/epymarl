REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .fpo_episode_runner import FPOEpisodeRunner
REGISTRY["fpo_episode"] = FPOEpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
