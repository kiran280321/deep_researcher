from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from retriever import Retriever
from typing import List, Dict
import textwrap
GEN_MODEL_NAME = "google/flan-t5-small"
class DeepResearcherAgent:
    def __init__(self, gen_model_name=GEN_MODEL_NAME, retriever: Retriever = None, device: int = -1):
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
        self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=device)
        self.retriever = retriever or Retriever()
    def plan_subtasks(self, question: str, max_subtasks: int = 4) -> List[str]:
        prompt = f"""Break the research query into up to {max_subtasks} concise sub-questions (one per line) that will help find relevant facts/evidence. Query: \"\"\"{question}\"\"\"
Return only sub-questions, numbered or bullet lines are fine."""
        res = self.pipeline(prompt, max_length=256)[0]["generated_text"].strip()
        lines = [l.strip(" -1234567890. ") for l in res.splitlines() if l.strip()]
        if len(lines) == 0:
            return [question]
        return lines
    def retrieve_for_subtask(self, subtask: str, top_k: int = 5):
        return self.retriever.query(subtask, top_k=top_k)
    def synthesize(self, question: str, retrieved_passages: List[Dict], reasoning_steps: List[str]) -> Dict:
        evidence_text = "\n\n".join([f"Source: {p['source']}\nExcerpt: {textwrap.shorten(p['text'], 400)}" for p in retrieved_passages])
        prompt = f"""You are a research assistant. The user question is: \"\"\"{question}\"\"\"\n\nUse the following evidence to construct a concise answer (3 sections):\n1) A short answer/summary\n2) Key evidence points (bullet list with source file names)\n3) Explanation of your reasoning steps (brief)\n\nEvidence:\n{evidence_text}\n\nAnswer:"""
        out = self.pipeline(prompt, max_length=512)[0]["generated_text"].strip()
        return {"answer": out, "evidence": retrieved_passages, "reasoning_steps": reasoning_steps}
    def research(self, question: str, top_k_per_subtask: int = 4, max_subtasks: int = 4):
        subtasks = self.plan_subtasks(question, max_subtasks=max_subtasks)
        all_retrieved = []
        reasoning_steps = []
        for st in subtasks:
            retrieved = self.retrieve_for_subtask(st, top_k=top_k_per_subtask)
            reasoning_steps.append(f"Subtask: {st} -> Retrieved {len(retrieved)} chunks")
            all_retrieved.extend(retrieved)
        seen = set()
        deduped = []
        for r in all_retrieved:
            key = (r["source"], r["chunk_id"])
            if key in seen: continue
            seen.add(key)
            deduped.append(r)
        deduped.sort(key=lambda x: x["score"], reverse=True)
        top_evidence = deduped[:12]
        synthesis = self.synthesize(question, top_evidence, reasoning_steps)
        return synthesis
