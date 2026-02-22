"""Integration tests of training and evaluation.

See testdata/data for code for regenerating data and accuracy/loss statistics.
"""

import contextlib
import difflib
import os
import re
import tempfile

import pytest

from yoyodyne.cli import main

DIR = os.path.relpath(os.path.dirname(__file__), os.getcwd())
CONFIG_DIR = os.path.join(DIR, "testdata/configs")
TESTDATA_DIR = os.path.join(DIR, "testdata/data")
CHECKPOINT_CONFIG_PATH = os.path.join(CONFIG_DIR, "checkpoint.yaml")
TOY_DATA = ["copy", "identity", "reverse", "upper"]
TOY_DATA_CONFIG_PATH = os.path.join(CONFIG_DIR, "toy_data.yaml")
TOY_TRAINER_CONFIG_PATH = os.path.join(CONFIG_DIR, "toy_trainer.yaml")
REAL_TRAINER_CONFIG_PATH = os.path.join(CONFIG_DIR, "real_trainer.yaml")
ARCH = [
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
    "soft_attention_gru",
    "soft_attention_lstm",
    "soft_attention_lstm_student_forcing",
    "soft_attention_lstm_gru_source",
    "soft_attention_lstm_transformer_source",
    "transformer",
    "transformer_student_forcing",
]
INFLECTION_ARCH = [
    "context_hard_attention_lstm_separate_features",
    "hard_attention_lstm_separate_features",
    "pointer_generator_lstm_separate_features",
    "soft_attention_lstm_gru_features",
    "soft_attention_lstm_linear_features",
    "soft_attention_lstm_separate_features",
    "soft_attention_lstm_shared_features",
    "transformer_invariant_features",
    "transformer_shared_features",
]
SEED = 49


class TestYoyodyne:

    @pytest.fixture(autouse=True)
    def setup_tempdir(self):
        """Replaces setUp and tearDown."""
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
        train_path = os.path.join(testdata_dir, "train.tsv")
        self.assertNonEmptyFileExists(train_path)
        dev_path = os.path.join(testdata_dir, "dev.tsv")
        self.assertNonEmptyFileExists(dev_path)
        test_path = os.path.join(testdata_dir, "test.tsv")
        self.assertNonEmptyFileExists(test_path)
        model_dir = os.path.join(self.tempdir.name, "models")
        model_config_path = os.path.join(CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        self.assertNonEmptyFileExists(trainer_config_path)
        main.python_interface(
            [
                "fit",
                f"--checkpoint={CHECKPOINT_CONFIG_PATH}",
                f"--data={data_config_path}",
                f"--data.train={train_path}",
                f"--data.val={dev_path}",
                f"--data.model_dir={model_dir}",
                f"--model={model_config_path}",
                f"--seed_everything={SEED}",
                f"--trainer={trainer_config_path}",
            ]
        )
        checkpoint_path = (
            f"{model_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
        )
        self.assertNonEmptyFileExists(checkpoint_path)
        evaluation_path = os.path.join(
            self.tempdir.name, f"{data}_{arch}.test"
        )
        with open(evaluation_path, "w") as sink:
            with contextlib.redirect_stdout(sink):
                main.python_interface(
                    [
                        "test",
                        f"--ckpt_path={checkpoint_path}",
                        f"--data={data_config_path}",
                        f"--data.test={test_path}",
                        f"--data.model_dir={model_dir}",
                        f"--model={model_config_path}",
                        "--trainer.enable_progress_bar=false",
                    ]
                )
        self.assertNonEmptyFileExists(evaluation_path)
        expected_path = os.path.join(
            TESTDATA_DIR, data, f"{arch}_expected.test"
        )
        self.assertFileIdentity(evaluation_path, expected_path)
        predicted_path = os.path.join(self.tempdir.name, f"{data}_{arch}.txt")
        main.python_interface(
            [
                "predict",
                f"--ckpt_path={checkpoint_path}",
                f"--data={data_config_path}",
                f"--data.model_dir={model_dir}",
                f"--data.predict={test_path}",
                f"--model={model_config_path}",
                f"--prediction.path={predicted_path}",
            ]
        )
        self.assertNonEmptyFileExists(predicted_path)
        expected_path = os.path.join(
            TESTDATA_DIR, data, f"{arch}_expected.txt"
        )
        self.assertFileIdentity(predicted_path, expected_path)

    # Misconfiguration tests.

    def test_misconfiguration_source_embedding_neq_model_embedding(self):
        self._test_misconfiguration_procedure(
            "ice_g2p",
            "misconfigured_source_embedding_neq_model_embedding",
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

    def test_misconfiguration_encoder_layers_neq_decoder_layers(self):
        self._test_misconfiguration_procedure(
            "ice_g2p",
            "misconfigured_encoder_layers_neq_decoder_layers",
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
        with pytest.raises(ValueError):
            main.python_interface(
                [
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
