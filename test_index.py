from src.retriever.retrieve import retrieve_context
hits = retrieve_context("definisi chiasma opticum", top_k=3)
print(hits)