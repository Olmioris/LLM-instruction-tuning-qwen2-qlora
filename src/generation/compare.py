def compare_generations(baseline_fn, finetuned_fn, prompts):
    results = []
    for p in prompts:
        base = baseline_fn(p)
        fine = finetuned_fn(p)
        results.append({"prompt": p, "baseline": base, "finetuned": fine})
    return results