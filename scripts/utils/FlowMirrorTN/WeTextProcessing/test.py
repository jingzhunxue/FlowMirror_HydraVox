from tn.chinese.normalizer import Normalizer



normalizer = Normalizer(cache_dir="/home/ecs-user/code/zeying/TTS_TN/flowTTS_TN/WeTextProcessing/cache")
result = normalizer.normalize("2.5平方电线")
print(result)