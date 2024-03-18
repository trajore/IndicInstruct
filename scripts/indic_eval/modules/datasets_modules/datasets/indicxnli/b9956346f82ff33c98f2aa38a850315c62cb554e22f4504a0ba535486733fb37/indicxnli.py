# coding=utf-8


# Lint as: python3
"""IndicXNLI: The Cross-Lingual NLI Corpus for Indic Languages."""


import os
import json

import datasets


_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2204.08776,
  doi = {10.48550/ARXIV.2204.08776},
  
  url = {https://arxiv.org/abs/2204.08776},
  
  author = {Aggarwal, Divyanshu and Gupta, Vivek and Kunchukuttan, Anoop},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {IndicXNLI: Evaluating Multilingual Inference for Indian Languages}, 
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
}"""

_DESCRIPTION = """\
IndicXNLI is a translated version of XNLI to 11 Indic Languages. As with XNLI, the goal is
to predict textual entailment (does sentence A imply/contradict/neither sentence
B) and is a classification task (given two sentences, predict one of three
labels).
"""

_LANGUAGES = (
    'hi',
    'bn',
    'mr',
    'as',
    'ta',
    'te',
    'or',
    'ml',
    'pa',
    'gu',
    'kn'
)


_URL = "https://huggingface.co/datasets/Divyanshu/indicxnli/resolve/main/forward"

class IndicxnliConfig(datasets.BuilderConfig):
    """BuilderConfig for XNLI."""

    def __init__(self, language: str, **kwargs):
        """BuilderConfig for XNLI.

        Args:
        language: One of hi, bn, mr, as, ta, te, or, ml, pa, gu, kn
          **kwargs: keyword arguments forwarded to super.
        """
        super(IndicxnliConfig, self).__init__(**kwargs)
        
        self.language = language
        self.languages = _LANGUAGES
        
        self._URLS = {
            "train": os.path.join(_URL, "train", f"xnli_{self.language}.json"),
            "test": os.path.join(_URL, "test", f"xnli_{self.language}.json"),
            "dev": os.path.join(_URL, "dev", f"xnli_{self.language}.json")
        }


class Indicxnli(datasets.GeneratorBasedBuilder):
    """IndicXNLI: The Cross-Lingual NLI Corpus for Indic Languages. Version 1.0."""

    VERSION = datasets.Version("1.0.0", "")
    BUILDER_CONFIG_CLASS = IndicxnliConfig
    BUILDER_CONFIGS = [
        IndicxnliConfig(
            name=lang,
            language=lang,
            version=datasets.Version("1.0.0", ""),
            description=f"Plain text import of IndicXNLI for the {lang} language",
        )
        for lang in _LANGUAGES
    ]

    def _info(self):
        features = datasets.Features(
            {
                "premise": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://github.com/divyanshuaggarwal/IndicXNLI",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = self.config._URLS
        
        
        downloaded_files = dl_manager.download(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "data_format": "IndicXNLI",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"], "data_format": "IndicXNLI"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"], "data_format": "IndicXNLI"},
            ),
        ]

    def _generate_examples(self, data_format, filepath):
        """This function returns the examples in the raw (text) form."""

        with open(filepath, "r") as f:
            data = json.load(f)
            data = data[list(data.keys())[0]]

        for idx, row in enumerate(data):
            yield idx, {
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row["label"],
            }
