"""药名归一化测试：折叠空间（全角/大小写/空格）与最长优先扫描。"""

from __future__ import annotations

import pytest

from harness_agent.safety import SEED_DRUG_DICTIONARY, DrugDictionary, DrugNormalizer, fold_text


@pytest.fixture()
def normalizer() -> DrugNormalizer:
    return DrugNormalizer(DrugDictionary(SEED_DRUG_DICTIONARY))


class TestFoldText:
    def test_fullwidth_lower_and_strip(self):
        assert fold_text("Ａ b　C d") == "abcd"

    def test_plain_text_passthrough(self):
        assert fold_text("阿莫西林") == "阿莫西林"

    def test_strip_zero_width_chars(self):
        """零宽字符必须剥离：否则 find_mentions 的 str.find 恒为 -1。"""
        assert fold_text("阿莫\u200b西林") == "阿莫西林"
        assert fold_text("阿莫\u200d西林") == "阿莫西林"
        assert fold_text("阿莫\u00ad西林") == "阿莫西林"
        assert fold_text("阿莫\ufeff西林") == "阿莫西林"

    def test_strip_infix_separators(self):
        """中缀分隔符必须剥离：否则「阿莫-西林」绕过全部三道闸门。"""
        assert fold_text("阿莫-西林") == "阿莫西林"
        assert fold_text("阿莫·西林") == "阿莫西林"
        assert fold_text("阿莫/西林") == "阿莫西林"

    def test_fullwidth_separator_is_stripped_after_conversion(self):
        """全角分隔符先转半角、再剥离（顺序敏感）。"""
        assert fold_text("ＡＭＯＸＩ－ＣＩＬＬＩＮ") == "amoxicillin"

    def test_stripping_is_symmetric_for_dictionary_and_text(self):
        """词典别名与待扫描文本共用 fold_text，两侧折叠结果必须一致。"""
        dictionary = DrugDictionary(SEED_DRUG_DICTIONARY)
        for alias in ("阿莫西林", "再林", "amoxicillin"):
            assert fold_text(alias) == fold_text(fold_text(alias))
            assert dictionary.alias_index[fold_text(alias)] == "amoxicillin"


class TestNormalize:
    def test_generic_name(self, normalizer):
        assert normalizer.normalize("阿莫西林") == "amoxicillin"

    def test_historical_alias(self, normalizer):
        assert normalizer.normalize("盘尼西林") == "penicillin"

    def test_brand_name(self, normalizer):
        assert normalizer.normalize("拜阿司匹灵") == "aspirin"

    def test_english_uppercase(self, normalizer):
        assert normalizer.normalize("AMOXICILLIN") == "amoxicillin"

    def test_fullwidth_english(self, normalizer):
        assert normalizer.normalize("ＣＥＦＴＲＩＡＸＯＮＥ") == "ceftriaxone"

    def test_whitespace_inside(self, normalizer):
        assert normalizer.normalize("头孢 曲松") == "ceftriaxone"

    def test_normalized_name_itself(self, normalizer):
        assert normalizer.normalize("penicillin") == "penicillin"

    def test_unknown_returns_none(self, normalizer):
        assert normalizer.normalize("不存在的药") is None


class TestFindMentions:
    def test_multiple_drugs_in_one_text(self, normalizer):
        mentions = normalizer.find_mentions("既往用过阿莫西林和头孢曲松")
        assert {m.normalized_name for m in mentions} == {"amoxicillin", "ceftriaxone"}

    def test_longest_match_first(self, normalizer):
        """benzylpenicillin 含子串 penicillin，长键优先只产出一条提及。"""
        mentions = normalizer.find_mentions("静滴 benzylpenicillin 80万U")
        assert [m.normalized_name for m in mentions] == ["penicillin"]

    def test_no_drugs(self, normalizer):
        assert normalizer.find_mentions("患者血常规未见明显异常") == []

    def test_empty_text(self, normalizer):
        assert normalizer.find_mentions("") == []

    def test_english_with_dosage(self, normalizer):
        mentions = normalizer.find_mentions("AMOXICILLIN 500mg bid")
        assert [m.normalized_name for m in mentions] == ["amoxicillin"]
