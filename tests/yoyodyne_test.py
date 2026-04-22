"""Integration tests of training and evaluation.

See testdata/data for code for regenerating data and accuracy/loss statistics.
"""

import contextlib
import difflib
import re
import os
import subprocess
import tempfile

import pytest

DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
CONFIG_DIR = os.path.join(DIR, "testdata/configs")
TESTDATA_DIR = os.path.join(DIR, "testdata/data")
CHECKPOINT_CONFIG_PATH = os.path.join(CONFIG_DIR, "checkpoint.yaml")
TOY_DATA = ["copy", "identity", "reverse", "upper"]
TOY_DATA_CONFIG_PATH = os.path.join(CONFIG_DIR, "toy_data.yaml")
TOY_TRAINER_CONFIG_PATH = os.path.join(CONFIG_DIR, "toy_trainer.yaml")
REAL_TRAINER_CONFIG_PATH = os.path.join(CONFIG_DIR, "real_trainer.yaml")
# Ones we'll test on any datasets except inflection.
ARCH = [
    "causal_transformer",
    "causal_transformer_student_forcing",
    "context_hard_attention_gru",
    "context_hard_attention_lstm",
    "gru",
    "hard_attention_gru",
    "hard_attention_lstm",
    "lstm",
    "pointer_generator_gru",
    "pointer_generator_lstm",
    "pointer_generator_lstm_student_forcing",
    "pointer_generator_transformer",
    "pointer_generator_transformer_student_forcing",
    "rotary_causal_transformer",
    "rotary_pointer_generator_transformer",
    "rotary_transformer",
    "soft_attention_gru",
    "soft_attention_lstm",
    "soft_attention_lstm_student_forcing",
    "soft_attention_lstm_gru_source",
    "soft_attention_lstm_transformer_source",
    "transformer",
    "transformer_absolute_positional",
    "transformer_student_forcing",
    "transducer_gru",
    "transducer_lstm",
    "transducer_lstm_student_forcing",
]
# Ones we'll test beam decoding on. We set aside student forcing and rotary
# positional encoding because they are orthogonal. No beam-capable arch is a
# transducer.
BEAM_ARCH = [
    "causal_transformer",
    "gru",
    "lstm",
    "pointer_generator_gru",
    "pointer_generator_lstm",
    "pointer_generator_transformer",
    "soft_attention_gru",
    "soft_attention_lstm",
    "transformer",
]
# Specific tests for the inflection data, which is larger and has features.
INFLECTION_ARCH = [
    "causal_transformer",
    "context_hard_attention_lstm_separate_features",
    "hard_attention_lstm_separate_features",
    "pointer_generator_lstm_separate_features",
    "pointer_generator_transformer_linear_features",
    "rotary_causal_transformer",
    "rotary_pointer_generator_transformer_linear_features",
    "rotary_transformer_shared_features",
    "soft_attention_lstm_gru_features",
    "soft_attention_lstm_linear_features",
    "soft_attention_lstm_separate_features",
    "soft_attention_lstm_shared_features",
    "transformer_invariant_features",
    "transformer_null_positional_features",
    "transformer_shared_features",
    "transducer_gru_linear_features",
    "transducer_lstm_linear_features",
]
SEED = 49

# Session-scoped state for checkpoint caching. Keyed by (data, arch,
# data_config_path, trainer_config_path).
_checkpoints: dict[tuple[str, str, str, str], str] = {}
_session_tempdir: tempfile.TemporaryDirectory | None = None


@pytest.fixture(scope="session", autouse=True)
def session_tempdir():
    """Creates a single tempdir for the session to hold trained checkpoints."""
    global _session_tempdir
    _session_tempdir = tempfile.TemporaryDirectory(prefix="yoyodyne_test-")
    yield
    _session_tempdir.cleanup()


def _get_or_train(
    data: str,
    arch: str,
    data_config_path: str,
    trainer_config_path: str,
) -> str:
    """Returns a checkpoint path for the given data/arch, training if needed.

    Checkpoints are cached for the duration of the test session so that
    test_toy and test_toy_beam can share the same trained model.

    Args:
        data (str).
        arch (str).
        data_config_path (str).
        trainer_config_path (str).

    Returns:
        str: path to the checkpoint.
    """
    key = (data, arch, data_config_path, trainer_config_path)
    if key in _checkpoints:
        return _checkpoints[key]
    testdata_dir = os.path.join(TESTDATA_DIR, data)
    train_path = os.path.join(testdata_dir, "train.tsv")
    dev_path = os.path.join(testdata_dir, "dev.tsv")
    if arch.startswith("transducer"):
        sed_path = os.path.join(testdata_dir, "train.sed")
        sed_args = [f"--model.sed_path={sed_path}"]
    else:
        sed_args = []
    # Include data and arch in the model dir to avoid collisions between
    # different (data, arch) pairs in the shared session tempdir.
    model_dir = os.path.join(_session_tempdir.name, "models", data, arch)
    model_config_path = os.path.join(CONFIG_DIR, f"{arch}.yaml")
    subprocess.check_call(
        [
            "yoyodyne",
            "fit",
            f"--checkpoint={CHECKPOINT_CONFIG_PATH}",
            f"--data={data_config_path}",
            f"--data.model_dir={model_dir}",
            f"--data.train={train_path}",
            f"--data.val={dev_path}",
            f"--model={model_config_path}",
            *sed_args,
            f"--seed_everything={SEED}",
            f"--trainer={trainer_config_path}",
        ]
    )
    checkpoint_path = (
        f"{model_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
    )
    _checkpoints[key] = checkpoint_path
    return checkpoint_path


class TestYoyodyne:

    @pytest.fixture(autouse=True)
    def setup_tempdir(self):
        """Creates a per-test tempdir for output files (predictions, etc.)."""
        self.tempdir = tempfile.TemporaryDirectory(prefix="yoyodyne_test-")
        yield
        self.tempdir.cleanup()

    def assertNonEmptyFileExists(self, path: str):
        assert os.path.exists(path), f"file {path} not found"
        assert os.stat(path).st_size > 0, f"file {path} is empty"

    def assertFileIdentity(self, actual_path: str, expected_path: str):
        with (
            open(actual_path, "r") as actual,
            open(expected_path, "r") as expected,
        ):
            difflines = "".join(
                difflib.unified_diff(
                    [self._normalize(line) for line in actual],
                    [self._normalize(line) for line in expected],
                    fromfile=actual_path,
                    tofile=expected_path,
                    n=1,
                )
            )
            if difflines:
                pytest.fail(f"Prediction differences found:\n{difflines}")

    @staticmethod
    def _normalize(line: str) -> str:
        return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", line)

    @pytest.mark.parametrize("data", TOY_DATA)
    @pytest.mark.parametrize("arch", ARCH)
    def test_toy(self, data: str, arch: str):
        self._test_model_procedure(
            data, arch, TOY_DATA_CONFIG_PATH, TOY_TRAINER_CONFIG_PATH
        )

    @pytest.mark.parametrize("data", TOY_DATA)
    @pytest.mark.parametrize("arch", BEAM_ARCH)
    def test_toy_beam(self, data: str, arch: str):
        self._test_beam_procedure(
            data, arch, TOY_DATA_CONFIG_PATH, TOY_TRAINER_CONFIG_PATH
        )

    @pytest.mark.parametrize("arch", ARCH)
    def test_ice_g2p(self, arch: str):
        data_config_path = os.path.join(CONFIG_DIR, "ice_g2p_data.yaml")
        self._test_model_procedure(
            "ice_g2p", arch, data_config_path, REAL_TRAINER_CONFIG_PATH
        )

    @pytest.mark.parametrize("arch", INFLECTION_ARCH)
    def test_tur_inflection(self, arch: str):
        data_config_path = os.path.join(CONFIG_DIR, "tur_inflection_data.yaml")
        self._test_model_procedure(
            "tur_inflection",
            arch,
            data_config_path,
            REAL_TRAINER_CONFIG_PATH,
        )

    def _test_model_procedure(
        self,
        data: str,
        arch: str,
        data_config_path: str,
        trainer_config_path: str,
    ):
        """Helper for test running."""
        testdata_dir = os.path.join(TESTDATA_DIR, data)
        test_path = os.path.join(testdata_dir, "test.tsv")
        self.assertNonEmptyFileExists(test_path)
        if arch.startswith("transducer"):
            sed_path = os.path.join(testdata_dir, "train.sed")
            self.assertNonEmptyFileExists(sed_path)
            sed_args = [f"--model.sed_path={sed_path}"]
        else:
            sed_args = []
        model_config_path = os.path.join(CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        self.assertNonEmptyFileExists(trainer_config_path)
        checkpoint_path = _get_or_train(
            data, arch, data_config_path, trainer_config_path
        )
        model_dir = os.path.join(_session_tempdir.name, "models", data, arch)
        self.assertNonEmptyFileExists(checkpoint_path)
        evaluation_path = os.path.join(
            self.tempdir.name, f"{data}_{arch}.test"
        )
        result = subprocess.run(
            [
                "yoyodyne",
                "test",
                f"--ckpt_path={checkpoint_path}",
                f"--data={data_config_path}",
                f"--data.test={test_path}",
                f"--data.model_dir={model_dir}",
                f"--model={model_config_path}",
                *sed_args,
                "--trainer.enable_progress_bar=false",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(evaluation_path, "w") as sink:
            sink.write(result.stdout)
        self.assertNonEmptyFileExists(evaluation_path)
        expected_path = os.path.join(
            TESTDATA_DIR, data, f"{arch}_expected.test"
        )
        self.assertFileIdentity(evaluation_path, expected_path)
        predicted_path = os.path.join(self.tempdir.name, f"{data}_{arch}.txt")
        subprocess.check_call(
            [
                "yoyodyne",
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--data={data_config_path}",
                f"--data.model_dir={model_dir}",
                f"--data.predict={test_path}",
                f"--model={model_config_path}",
                *sed_args,
                f"--prediction.path={predicted_path}",
            ]
        )
        self.assertNonEmptyFileExists(predicted_path)
        expected_path = os.path.join(
            TESTDATA_DIR, data, f"{arch}_expected.txt"
        )
        self.assertFileIdentity(predicted_path, expected_path)

    def _test_beam_procedure(
        self,
        data: str,
        arch: str,
        data_config_path: str,
        trainer_config_path: str,
    ):
        """Helper for beam decoding tests."""
        testdata_dir = os.path.join(TESTDATA_DIR, data)
        test_path = os.path.join(testdata_dir, "test.tsv")
        self.assertNonEmptyFileExists(test_path)
        # No beam-capable arch is a transducer, so no sed_args needed.
        model_config_path = os.path.join(CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        self.assertNonEmptyFileExists(trainer_config_path)
        checkpoint_path = _get_or_train(
            data, arch, data_config_path, trainer_config_path
        )
        model_dir = os.path.join(_session_tempdir.name, "models", data, arch)
        self.assertNonEmptyFileExists(checkpoint_path)
        predicted_path = os.path.join(
            self.tempdir.name, f"{data}_{arch}_beam.txt"
        )
        subprocess.check_call(
            [
                "yoyodyne",
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--data={data_config_path}",
                f"--data.model_dir={model_dir}",
                f"--data.predict={test_path}",
                f"--model={model_config_path}",
                "--model.beam_width=5",
                f"--prediction.path={predicted_path}",
            ]
        )
        self.assertNonEmptyFileExists(predicted_path)
        expected_path = os.path.join(
            TESTDATA_DIR, data, f"{arch}_beam_expected.txt"
        )
        self.assertFileIdentity(predicted_path, expected_path)

    # Misconfiguration tests.

    def test_misconfiguration_encoder_layers_neq_decoder_layers(self):
        self._test_misconfiguration_procedure(
            "ice_g2p",
            "misconfigured_encoder_layers_neq_decoder_layers",
        )

    def test_misconfiguration_features_embedding_neq_model_embedding(self):
        self._test_misconfiguration_procedure(
            "tur_inflection",
            "misconfigured_features_embedding_neq_model_embedding",
        )

    def test_misconfiguration_features_col_no_features_encoder(self):
        self._test_misconfiguration_procedure(
            "tur_inflection",
            "soft_attention_lstm",
        )

    def test_misconfiguration_features_encoder_no_features_col(self):
        self._test_misconfiguration_procedure(
            "ice_g2p",
            "soft_attention_lstm_shared_features",
        )

    def test_misconfiguration_source_embedding_neq_model_embedding(self):
        self._test_misconfiguration_procedure(
            "ice_g2p",
            "misconfigured_source_embedding_neq_model_embedding",
        )

    def _test_misconfiguration_procedure(self, data: str, arch: str):
        testdata_dir = os.path.join(TESTDATA_DIR, data)
        train_path = os.path.join(testdata_dir, "train.tsv")
        self.assertNonEmptyFileExists(train_path)
        dev_path = os.path.join(testdata_dir, "dev.tsv")
        model_dir = os.path.join(self.tempdir.name, "models")
        data_config_path = os.path.join(CONFIG_DIR, f"{data}_data.yaml")
        model_config_path = os.path.join(CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(
                [
                    "yoyodyne",
                    "validate",
                    f"--checkpoint={CHECKPOINT_CONFIG_PATH}",
                    f"--data={data_config_path}",
                    f"--model={model_config_path}",
                    f"--trainer={TOY_TRAINER_CONFIG_PATH}",
                    f"--data.train={train_path}",
                    f"--data.val={dev_path}",
                    f"--data.model_dir={model_dir}",
                ]
            )
