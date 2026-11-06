"""Compatibility entry point for the orchestrator already running in memory.
Remove this entry point after that legacy run ends.
"""
if __name__ == "__main__":
    print("SKIPPED: catalog-only pipeline. Continue to export_embeddings.", flush=True)
