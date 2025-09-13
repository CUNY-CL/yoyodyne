"""Full tests of training and prediction.

This runs five epochs of training over toy data sets, then computes the
held-out accuracy performance. As such this is essentially a
change-detector test.
"""

import contextlib
import difflib
import itertools
import os
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
            diff = list(
                difflib.unified_diff(
                    actual.readlines(),
                    expected.readlines(),
                    fromfile=actual_path,
                    tofile=expected_path,
                    n=1,
                )
            )
        self.assertEqual(diff, [], f"Prediction differences found:\n{diff}")

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
        "attentive_gru",
        "attentive_lstm",
        "attentive_lstm_gru_source",
        "attentive_lstm_transformer_source",
        "context_hard_attention_gru",
        "context_hard_attention_lstm",
        "gru",
        "hard_attention_gru",
        "hard_attention_lstm",
        "lstm",
        "pointer_generator_gru",
        "pointer_generator_lstm",
        "pointer_generator_transformer",
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

    # Also tests out separate features encoders.
    INFLECTION_ARCH = [
        "attentive_lstm_gru_features",
        "attentive_lstm_linear_features",
        "attentive_lstm_separate_features",
    ]

    @parameterized.expand(ARCH + INFLECTION_ARCH)
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

    def test_misconfiguration_source_embedding_neq_model_embedding(self):
        self._test_misconfiguration_procedure(
            "misconfigured_source_embedding_neq_model_embedding",
        )

    def test_misconfiguration_features_embedding_neq_model_embedding(self):
        self._test_misconfiguration_procedure(
            "misconfigured_features_embedding_neq_model_embedding",
        )

    def test_misconfiguration_encoder_layers_neq_decoder_layers(self):
        self._test_misconfiguration_procedure(
            "misconfigured_encoder_layers_neq_decoder_layers",
        )

    def _test_misconfiguration_procedure(self, arch: str):
        # It doesn't really matter what data we use, though.
        testdata_dir = os.path.join(self.TESTDATA_DIR, "tur_inflection")
        train_path = os.path.join(testdata_dir, "train.tsv")
        self.assertNonEmptyFileExists(train_path)
        dev_path = os.path.join(testdata_dir, "dev.tsv")
        model_dir = os.path.join(self.tempdir.name, "models")
        # Gets config paths.
        model_config_path = os.path.join(self.CONFIG_DIR, f"{arch}.yaml")
        self.assertNonEmptyFileExists(model_config_path)
        # Tries to fit but catches the misconfiguration.
        # Ideally we'd catch the more specific example but jsonargparse
        # catches these and turns them into ValueError.
        with self.assertRaises(ValueError):
            main.python_interface(
                [
                    "fit",
                    f"--checkpoint={self.CHECKPOINT_CONFIG_PATH}",
                    f"--data={self.TOY_DATA_CONFIG_PATH}",
                    f"--model={model_config_path}",
                    f"--trainer={self.TOY_TRAINER_CONFIG_PATH}",
                    f"--data.train={train_path}",
                    f"--data.val={dev_path}",
                    f"--data.model_dir={model_dir}",
                    f"--seed_everything={self.SEED}",
                ]
            )


if __name__ == "__main__":
    unittest.main()
