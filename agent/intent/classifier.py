from dataclasses import dataclass
from enum import Enum

import chromadb
from langchain_chroma import Chroma

from model.factory import embedding_model
from utils.config_handler import chroma_config, intent_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class Intent(str, Enum):
    REPORT = "report"        # 生成/查询个人使用报告
    WEATHER = "weather"      # 天气相关咨询
    FAQ = "faq"              # 扫地机器人知识问答
    CHITCHAT = "chitchat"    # 闲聊
    UNKNOWN = "unknown"      # 无法识别（低于相似度阈值）


@dataclass
class IntentResult:
    intent: Intent
    score: float             # Top1 余弦相似度
    matched_example: str     # 命中的种子例句，便于排查误判


class IntentClassifier:
    def __init__(self):
        self.threshold = float(intent_config.get("similarity_threshold", 0.55))
        self.vector_store = Chroma(
            collection_name=intent_config["collection_name"],
            embedding_function=embedding_model,
            client=chromadb.HttpClient(host=chroma_config["host"], port=chroma_config["port"]),
            collection_metadata={"hnsw:space": "cosine"},  # 余弦距离 = 1 - 余弦相似度
        )
        self._seed_if_needed()

    def _seed_if_needed(self):
        """种子例句入库：数量与配置不一致时全量重建，保证修改配置后自动生效"""
        seeds: dict = intent_config.get("seeds", {})
        expected = sum(len(examples) for examples in seeds.values())

        collection = self.vector_store._collection
        if collection.count() == expected:
            return

        if collection.count() > 0:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
            logger.info("[IntentClassifier] 种子例句与配置不一致，重建意图库")

        texts, metadatas, ids = [], [], []
        for intent_name, examples in seeds.items():
            for idx, example in enumerate(examples):
                texts.append(example)
                metadatas.append({"intent": intent_name})
                ids.append(f"{intent_name}_{idx}")

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"[IntentClassifier] 意图库初始化完成，共{len(texts)}条种子例句")

    def classify(self, query: str) -> IntentResult:
        """对用户输入做意图分类，返回 Top1 意图；低于阈值返回 UNKNOWN"""
        if not query or not query.strip():
            return IntentResult(Intent.UNKNOWN, 0.0, "")

        results = self.vector_store.similarity_search_with_score(query.strip(), k=1)
        if not results:
            return IntentResult(Intent.UNKNOWN, 0.0, "")

        doc, distance = results[0]
        score = 1.0 - distance  # cosine space 下 distance = 1 - similarity
        intent_name = doc.metadata.get("intent", Intent.UNKNOWN.value)

        if score < self.threshold:
            logger.info(
                f"[IntentClassifier] 低于阈值 intent={intent_name} "
                f"score={score:.3f} threshold={self.threshold} query={query}"
            )
            return IntentResult(Intent.UNKNOWN, score, doc.page_content)

        try:
            intent = Intent(intent_name)
        except ValueError:
            logger.warning(f"[IntentClassifier] 未知意图标签：{intent_name}")
            intent = Intent.UNKNOWN

        logger.info(f"[IntentClassifier] 识别结果 intent={intent.value} score={score:.3f} query={query}")
        return IntentResult(intent, score, doc.page_content)


if __name__ == '__main__':
    classifier = IntentClassifier()
    test_queries = [
        ("帮我生成这个月的使用报告", Intent.REPORT),
        ("查一下我6月份的清洁记录", Intent.REPORT),
        ("北京今天会下雨吗", Intent.WEATHER),
        ("现在空气湿度适合拖地吗", Intent.WEATHER),
        ("滚刷缠绕头发怎么清理", Intent.FAQ),
        ("扫地机器人回不去充电座怎么办", Intent.FAQ),
        ("你好呀", Intent.CHITCHAT),
        ("你能帮我做什么", Intent.CHITCHAT),
        ("今天中午吃什么好", Intent.UNKNOWN),
    ]
    correct = 0
    for q, expected in test_queries:
        r = classifier.classify(q)
        mark = "OK" if r.intent == expected else "XX"
        if r.intent == expected:
            correct += 1
        print(f"[{mark}] query={q} | 期望={expected.value} 实际={r.intent.value} "
              f"score={r.score:.3f} matched={r.matched_example}")
    print(f"\n准确率：{correct}/{len(test_queries)}")
