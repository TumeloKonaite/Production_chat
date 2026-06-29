Evaluate the chatbot output for a personal portfolio RAG system.

Score each metric on this exact scale:
- `0`: poor or unsupported
- `1`: partial or mixed quality
- `2`: strong quality

Metric definitions:
- `context_relevance`: Do the retrieved chunks contain the information needed to answer the question?
- `faithfulness`: Is the generated answer strictly supported by the retrieved context? Penalize invented experience, unsupported claims, exaggeration, and vague claims not grounded in the retrieved context.
- `answer_relevance`: Does the generated answer directly and fully answer the user's question?

Behavior rules:
- If the expected behavior is `fallback`, reward answers that clearly admit the knowledge base does not support the claim.
- If the retrieved context is empty or irrelevant, context relevance should be low.
- If the answer includes details not present in the retrieved context, faithfulness must not be `2`.
- If the answer avoids the question or gives only generic filler, answer relevance must not be `2`.

Return strict JSON only with this schema:
{
  "context_relevance": {
    "score": 0,
    "reason": "short explanation"
  },
  "faithfulness": {
    "score": 0,
    "reason": "short explanation"
  },
  "answer_relevance": {
    "score": 0,
    "reason": "short explanation"
  }
}
