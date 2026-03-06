# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import html
import re
import urllib.parse as ul
from typing import Dict, List, Optional, Tuple, Union

import warnings

import ftfy
import torch
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from transformers import T5EncoderModel, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

DATA_TYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py
class CaptionProcessor:
    """
    Processes captions to remove unwanted characters and format them to use training and sampling.
    """

    def __init__(self) -> None:
        self.bad_punct_regex = re.compile(
            r"["
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)"
            + r"\("
            + r"\]"
            + r"\["
            + r"\}"
            + r"\{"
            + r"\|"
            + "\\"
            + r"\/"
            + r"\*"
            + r"]{1,}"
        )  # noqa

    @staticmethod
    def basic_clean(text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption: str) -> str:
        return self.clean_caption_logic(self.clean_caption_logic(caption))

    def clean_caption_logic(self, caption: str) -> str:
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)
        return caption.strip()


# Modified from: https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/models/utils.py
def text_encoder_embedding_format(enc: str) -> Tuple[int, int]:
    """Returns sequence length and token embedding dimension for text encoder."""
    if enc in ["google/flan-t5-large"]:
        return 120, 1024
    if enc in ["meta-llama/Llama-3.2-1B"]:
        return 256, 2048
    raise ValueError(f"Please specifcy the sequence and embedding size of {enc} encoder")


# Modified from: https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/models/utils.py
class UniversalTextEncoder(torch.nn.Module):
    """Universal text encoder supporting multiple model types.

    Args:
        name (str): Name/path of the model to load
        dtype (str): Data type for model weights
        pretrained (bool, True): Whether to load pretrained weights
    """

    def __init__(self, name: str, dtype: str, pretrained: bool = True):
        super().__init__()
        self.name = name
        if self.name == "google/flan-t5-large":
            self.encoder = T5EncoderModel.from_pretrained(
                name,
                torch_dtype=DATA_TYPES[dtype],
            )
        if self.name == "meta-llama/Llama-3.2-1B":
            self.encoder = AutoModelForCausalLM.from_pretrained(
                name, 
                torch_dtype=DATA_TYPES[dtype],
            )

    def encode(
        self,
        tokenized_caption: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.name == "google/flan-t5-large":
            out = self.encoder(tokenized_caption, attention_mask=attention_mask)["last_hidden_state"]
            out = out.unsqueeze(dim=1)
            return out, None
        if self.name == "meta-llama/Llama-3.2-1B":
            out = self.encoder(tokenized_caption, attention_mask=attention_mask, output_hidden_states=True)["hidden_states"][-1]
            out = out.unsqueeze(dim=1)
            return out, None


# Modified from: https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/models/utils.py
class UniversalTokenizer:
    """Universal tokenizer supporting multiple model types.

    Args:
        name (str): Name/path of the tokenizer to load
    """

    def __init__(self, name: str):
        self.name = name
        s, d = text_encoder_embedding_format(name)
        if self.name == "google/flan-t5-large":
            self.tokenizer = T5Tokenizer.from_pretrained(name)  # for t5 we would use a smaller than max_seq_length
        elif self.name == "meta-llama/Llama-3.2-1B":
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_max_length = s

    def tokenize(self, captions: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if self.name == "google/flan-t5-large":
            text_tokens_and_mask = self.tokenizer(
                captions,
                padding="max_length",
                max_length=self.model_max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            return {
                "input_ids": text_tokens_and_mask["input_ids"],
                "attention_mask": text_tokens_and_mask["attention_mask"],
            }
        elif self.name == "meta-llama/Llama-3.2-1B":
            text_tokens_and_mask = self.tokenizer(
                captions,
                padding="max_length",
                max_length=self.model_max_length,
                return_tensors="pt",
                truncation=True,
            )
            return {
                "input_ids": text_tokens_and_mask["input_ids"],
                "attention_mask": text_tokens_and_mask["attention_mask"],
            }
