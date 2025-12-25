"""
CosyVoice 包兼容层。

本仓库中 CosyVoice 代码位于 `server/model_utils/cosyvoice`，在运行时通常以
`server.model_utils.cosyvoice` 的包名被导入；但预训练配置（hyperpyyaml）里使用的是
`cosyvoice.*` 的绝对导入路径，例如：

    !new:cosyvoice.llm.llm_multi_head_v3.CosyVoice3LM

为了让上述路径在不改 YAML、不改 sys.path 的前提下可用，这里将当前包注册为
`cosyvoice` 的别名。
"""

import sys as _sys

# 当以 `server.model_utils.cosyvoice` 导入时，给出 `cosyvoice` 顶层别名
if __name__ != "cosyvoice":
    _sys.modules.setdefault("cosyvoice", _sys.modules[__name__])


