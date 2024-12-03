# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import os

from .constants import (
    CMUDICT,
    JIEBA_FREQ,
    JIEBA_POS,
    UNIVERSAL_POS,
    HKCANCOR_POS,
    STRICT_INITIALS,
    INITIALS,
    POSTNASALS,
    STRICT_FINALS,
    FINALS,
    TONES,
    ONSETS,
    NUCLEI,
    CODAS,
    JYUT_TONES,
    JYUT_FINALS,
    STRESS,
    PHONES,
)
from .g2p_mix import G2pMix
from .utils import load_dict

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# load dict to fix some badcase of pypinyin
load_dict()

__all__ = [
    "CMUDICT",
    "JIEBA_FREQ",
    "JIEBA_POS",
    "UNIVERSAL_POS",
    "HKCANCOR_POS",
    "STRICT_INITIALS",
    "INITIALS",
    "POSTNASALS",
    "STRICT_FINALS",
    "FINALS",
    "TONES",
    "ONSETS",
    "NUCLEI",
    "CODAS",
    "JYUT_TONES",
    "JYUT_FINALS",
    "STRESS",
    "PHONES",
    "G2pMix",
]
