"""Integration tests of training and evaluation.

See testdata/data for code for regenerating data and accuracy/loss statistics.
"""

import contextlib
import difflib
import itertools
import os
import re
import tempfile
import unittest

from parameterized import parameterized

from yoyodyne.cli import main


class YoyodyneTest(unittest.TestCase):
    def assertNonEmptyFileExists(self, path: str):
        self.assertTrue(os.path.exists(path), msg=f"file {path} not found")
        self.assertGreater(
            os.stat(path).st_size, 0, msg="file {path} is empty"
        )

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
                self.fail(f"Prediction differences found:\n{difflines}")

    @staticmethod
    def _normalize(line: str) -> str:
        return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", line)

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="yoyodyne_test-")

    def tearDown(self):
        self.tempdir.cleanup()

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
        "pointer_generator_transformer",
        "soft_attention_gru",
        "soft_attention_lstm",
        "soft_attention_lstm_gru_source",
        "soft_attention_lstm_transformer_source",
        "transformer",
        # TODO: test transducers too; but we need logic for the SED data.
    ]
    SEED = 49

    @parameterized.expand(itertools.product(TOY_DATA, ARCH))
    def test_toy(self, data: str, arch: str):
        self._test_model_procedure(
            data, arch, self.TOY_DATA_CONFIG_PATH, self.TOY_TRAINER_CONFIG_PATH
        )

    @parameterized.expand(ARCH)
    def test_g2p(self, arch: str):
        data_config_path = os.path.join(self.CONFIG_DIR, "ice_g2p_data.yaml")
        self._test_model_procedure(
            "ice_g2p", arch, data_config_path, self.REAL_TRAINER_CONFIG_PATH
        )

    # Also tests out different ways to encode the features.
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

    @parameterized.expand(INFLECTION_ARCH)
    def test_inflection(self, arch: str):
        data_config_path = os.path.join(
            self.CONFIG_DIR, "tur_inflection_data.yaml"
        )
        self._test_model_procedure(
            "tur_inflection",
            arch,
            data_config_path,
            self.REAL_TRAINER_CONFIG_PATH,
        )

    def _test_model_procedure(
        self,
        data: str,
        arch: str,
        data_config_path: str,
        trainer_config_path: str,
    ):
        """Helper for test running."""
        # Gets data paths.
        testdata_dir = os.path.join(self.TESTDATA_DIR, data)
        train_path = os.path.join(testdata_dir, "train.tsv")
        self.assertNonEmptyFileExists(train_path)
        dev_path = os.path.join(testdata_dir, "dev.tsv")
        self.assertNonEmptyFileExists(dev_path)
        test_path = os.path.join(testdata_dir, "test.tsv")
        self.assertNonEmptyFileExists(test_path)
        model_dir = os.path.join(self.tempdir.name, "models")
        # Gets config paths.
        model_config_path = os.path.join(self.CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        self.assertNonEmptyFileExists(trainer_config_path)
        # Fits and confirms creation of the checkpoint.
        main.python_interface(
            [
                "fit",
                f"--checkpoint={self.CHECKPOINT_CONFIG_PATH}",
                f"--data={data_config_path}",
                f"--data.train={train_path}",
                f"--data.val={dev_path}",
                f"--data.model_dir={model_dir}",
                f"--model={model_config_path}",
                f"--seed_everything={self.SEED}",
                f"--trainer={trainer_config_path}",
            ]
        )
        checkpoint_path = (
            f"{model_dir}/lightning_logs/version_0/checkpoints/last.ckpt"
        )
        self.assertNonEmptyFileExists(checkpoint_path)
        # Predicts on test data.
        predicted_path = os.path.join(
            self.tempdir.name, f"{data}_{arch}_predicted.txt"
        )
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
        evaluation_path = os.path.join(
            self.tempdir.name, f"{data}_{arch}_evaluated.test"
        )
        # Evaluates on test data and compares with result.
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
            self.TESTDATA_DIR, data, f"{arch}_expected.test"
        )
        self.assertFileIdentity(evaluation_path, expected_path)

    # These misconfiguration tests all use a shared structure.

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
        # It doesn't really matter what data we use, though.
        testdata_dir = os.path.join(self.TESTDATA_DIR, data)
        train_path = os.path.join(testdata_dir, "train.tsv")
        self.assertNonEmptyFileExists(train_path)
        dev_path = os.path.join(testdata_dir, "dev.tsv")
        model_dir = os.path.join(self.tempdir.name, "models")
        # Gets config paths.
        data_config_path = os.path.join(self.CONFIG_DIR, f"{data}_data.yaml")
        model_config_path = os.path.join(self.CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        # Tries to validate but catches the misconfiguration.
        # Errors during the constructor are converted to ValueError;
        # those during runtime inference are a more derived class.
        with self.assertRaises(ValueError):
            main.python_interface(
                [
                    "validate",
                    f"--checkpoint={self.CHECKPOINT_CONFIG_PATH}",
                    f"--data={data_config_path}",
                    f"--model={model_config_path}",
                    f"--trainer={self.TOY_TRAINER_CONFIG_PATH}",
                    f"--data.train={train_path}",
                    f"--data.val={dev_path}",
                    f"--data.model_dir={model_dir}",
                ]
            )


if __name__ == "__main__":
    unittest.main()
