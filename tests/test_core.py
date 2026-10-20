import unittest

from ollamamaker.core import recommended_context, recommended_output_limit, recommended_tiers
from ollamamaker.models import ContextPolicy, ModelArch


def model_arch(*, context_length: int = 262_144, output_architecture: str = "qwen3") -> ModelArch:
    return ModelArch(
        name="test:latest",
        architecture=output_architecture,
        context_length=context_length,
        num_layers=24,
        num_kv_heads=4,
        head_dim=128,
        num_ctx_override=0,
        parameter_count=3_000_000_000,
        quantization_level="Q4_K_M",
        model_weight_mib=2_048,
    )


class RecommendedContextTests(unittest.TestCase):
    def test_balanced_profile_stops_at_64k_when_more_context_fits(self) -> None:
        chosen = recommended_context(model_arch(), 24_576, 65_536)

        self.assertEqual(chosen, 65_536)

    def test_long_profile_targets_128k(self) -> None:
        chosen = recommended_context(model_arch(), 24_576, 65_536, policy=ContextPolicy(profile="long"))

        self.assertEqual(chosen, 131_072)

    def test_maximum_profile_is_explicit(self) -> None:
        chosen = recommended_context(model_arch(), 24_576, 65_536, policy=ContextPolicy(profile="maximum"))

        self.assertEqual(chosen, 262_144)

    def test_context_target_is_clamped_to_highest_gpu_safe_value(self) -> None:
        arch = model_arch()
        chosen = recommended_context(arch, 4_096, 65_536, policy=ContextPolicy(context_cap=131_072))
        tiers = recommended_tiers(arch, 4_096, 65_536)
        gpu_tiers = [ctx for ctx, _, placement in tiers if placement == "GPU-only"]

        self.assertEqual(chosen, gpu_tiers[-1])

    def test_parallel_requests_reduce_the_safe_context(self) -> None:
        arch = model_arch()
        single = recommended_context(arch, 4_096, 65_536, policy=ContextPolicy(profile="maximum"))
        parallel = recommended_context(
            arch,
            4_096,
            65_536,
            policy=ContextPolicy(profile="maximum", parallel=4),
        )

        self.assertLess(parallel, single)

    def test_never_recommends_context_that_requires_system_ram(self) -> None:
        chosen = recommended_context(model_arch(), 0, 65_536, policy=ContextPolicy(profile="maximum"))

        self.assertEqual(chosen, 0)


class RecommendedOutputLimitTests(unittest.TestCase):
    def test_balanced_output_is_capped_at_8k(self) -> None:
        chosen = recommended_output_limit(model_arch(), 65_536)

        self.assertEqual(chosen, 8_192)

    def test_output_never_uses_more_than_quarter_of_context(self) -> None:
        chosen = recommended_output_limit(model_arch(), 16_384, output_cap=32_768)

        self.assertEqual(chosen, 4_096)

    def test_architecture_output_limit_is_respected(self) -> None:
        arch = model_arch(output_architecture="llama")
        chosen = recommended_output_limit(arch, 65_536, profile="maximum")

        self.assertEqual(chosen, 4_096)


if __name__ == "__main__":
    unittest.main()
