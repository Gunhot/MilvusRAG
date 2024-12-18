from sentence_transformers import SentenceTransformer

# 모델 다운로드 및 로컬에 저장
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
model.save('./local_multilingual_e5_large_instruct')

