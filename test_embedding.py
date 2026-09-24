from embeddings.embedder import Embedder

embedder = Embedder()

query = "How does LoRA reduce trainable parameters?"

relevant = (
    "LoRA reduces the number of trainable parameters "
    "by introducing low-rank trainable matrices."
)

irrelevant = (
    "The Transformer architecture uses self-attention "
    "mechanisms to process sequences in parallel."
)

query_embedding = embedder.embed_query(query)

relevant_embedding = embedder.embed_documents([relevant])[0]
irrelevant_embedding = embedder.embed_documents([irrelevant])[0]

relevant_score = query_embedding @ relevant_embedding
irrelevant_score = query_embedding @ irrelevant_embedding

print("Relevant score:", relevant_score)
print("Irrelevant score:", irrelevant_score)