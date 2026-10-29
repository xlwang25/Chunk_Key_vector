"""Resolve the single in-process model; no remote workers are inspected."""
def resolve_capture_model(llm):
    engine = llm.llm_engine
    paths = [getattr(engine, 'model_executor', None)]
    core = getattr(getattr(engine, 'engine_core', None), 'engine_core', None)
    if core is not None:
        paths.append(getattr(core, 'model_executor', None))
    for executor in paths:
        worker = getattr(executor, 'driver_worker', None)
        for owner in (worker, getattr(worker, 'worker', None)):
            runner = getattr(owner, 'model_runner', None)
            model = getattr(runner, 'model', None)
            if model is not None and hasattr(model, 'set_collect_kv'):
                return model
    raise RuntimeError('KeyChunk requires patched vLLM with an in-process worker')
