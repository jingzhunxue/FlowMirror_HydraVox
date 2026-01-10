import requests
import os
import logging
from typing import Dict, Any, Optional, List
from ..i18n import t

logger = logging.getLogger(__name__)

class APIClient:
    """API 客户端，用于与后端服务通信"""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """通用请求方法"""
        url = f"{self.base_url}/api/v1{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url}, error: {e}")
            return {
                "success": False,
                "message": t("请求失败: {error}", error=str(e)),
                "error": str(e)
            }
    
    # =================== 训练相关 API ===================
    
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """启动训练任务"""
        return self._request("POST", "/training/start", json=config)
    
    def stop_training(self, training_id: str) -> Dict[str, Any]:
        """停止训练任务"""
        return self._request("POST", f"/training/stop/{training_id}")
    
    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """获取训练状态"""
        return self._request("GET", f"/training/status/{training_id}")
    
    def get_all_trainings(self) -> Dict[str, Any]:
        """获取所有训练任务"""
        return self._request("GET", "/training/list")
    
    def delete_training(self, training_id: str) -> Dict[str, Any]:
        """删除训练任务"""
        return self._request("DELETE", f"/training/{training_id}")
    
    # =================== TTS 相关 API ===================
    
    def text_to_speech(self, text: str, speaker_id: str, **kwargs) -> Dict[str, Any]:
        """文本转语音"""
        data = {
            "text": text,
            "speaker_id": speaker_id,
            **kwargs
        }
        return self._request("POST", "/tts", json=data)
    
    def zero_shot_tts(self, tts_text: str, prompt_text: str, prompt_audio_base64: str, **kwargs) -> Dict[str, Any]:
        """零样本语音合成"""
        data = {
            "tts_text": tts_text,
            "prompt_text": prompt_text,
            "prompt_audio_base64": prompt_audio_base64,
            **kwargs
        }
        return self._request("POST", "/zero-shot", json=data)
    
    def get_speakers(self) -> Dict[str, Any]:
        """获取说话人列表"""
        return self._request("GET", "/speakers")
    
    def load_model_weights(self, llm_pt: str, flow_pt: str) -> Dict[str, Any]:
        """加载模型权重"""
        data = {
            "llm_pt": llm_pt,
            "flow_pt": flow_pt
        }
        return self._request("POST", "/load_pt", json=data)

# 全局 API 客户端实例
api_client = APIClient()
