from pathlib import Path

import pytest

from project_llm.tokenizer.trainer import Trainer as TrainerType


@pytest.fixture(scope="class")
def trainer() -> TrainerType:
    return TrainerType()


class TestTokenizerTrainerType:
    _mock_decoder = {i: bytes([i]) for i in range(256)}
    _mock_merges = {(1, 1): 257}

    def test_train(self, trainer: TrainerType, tmp_path: Path) -> None:
        pth = Path(
            trainer.train(
                text="abcdefghiklmnop",
                vocab_size=257,
                fp=str(tmp_path / "tokenizer.bpe"),
            )
        )
        assert pth.exists()
        assert pth.read_text()

    def test_train_with_existing_file_exits(
        self, trainer: TrainerType, tmp_path: Path
    ) -> None:
        p = tmp_path / "exists"
        p.touch()
        with pytest.raises(SystemExit) as e:
            trainer.train("abcdefghijk", 1000, fp=str(p))
        assert e.value.code == 1

    def test_decoder_to_textlines(self, trainer: TrainerType) -> None:
        textlines = trainer.decoder_to_textlines(self._mock_decoder)
        assert len(textlines) == len(self._mock_decoder)

    def test_save(self, trainer: TrainerType, tmp_path: Path) -> None:
        p = tmp_path / "tokenizer.bpe"
        p.touch()
        respath = trainer.save(
            str(p), decoder=self._mock_decoder, merges=self._mock_merges
        )
        txt = p.read_text()
        assert str(p) == respath
        assert "[vocab]" in txt
        assert "[merges]" in txt
