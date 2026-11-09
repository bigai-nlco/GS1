"""One serving argument contract for the CLI and replica supervisor."""

from pathlib import Path

from gs1.api.settings import Settings

FIELDS = (
    "host",
    "port",
    "max_context",
    "max_questions",
    "max_request_tokens",
    "max_branches",
    "max_attention_bytes",
    "max_logit_elements",
    "batch_tokens",
    "max_queue",
    "deadline_seconds",
    "cache_mb",
    "graph_cache_mb",
    "graph_cache_entries",
    "cpu_threads",
)
FLAGS = {"cuda_graphs": "--experimental-cuda-graphs"}


def add_serving_arguments(parser):
    defaults = Settings()
    parser.add_argument("--model", "--model-path", dest="artifact", required=True)
    parser.add_argument("--model-name", dest="local_model", default=defaults.local_model)
    for name in FIELDS:
        value = getattr(defaults, name)
        parser.add_argument(
            "--" + name.replace("_", "-"), type=Settings.model_fields[name].annotation, default=value
        )
    parser.add_argument("--engine", choices=["reference", "shared"], default=defaults.engine)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default=defaults.precision)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--graph-warmup-requests", type=Path)
    parser.add_argument("--graph-capture-policy", choices=["prewarm", "online"], default="prewarm")
    parser.add_argument("--graph-slots", type=int, choices=[1, 2, 4], default=1)
    for name, flag in FLAGS.items():
        parser.add_argument(flag, dest=name, action="store_true", default=getattr(defaults, name))


def serving_settings(args):
    return Settings(**{name: getattr(args, name) for name in Settings.model_fields if hasattr(args, name)})
