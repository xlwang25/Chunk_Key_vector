"""Collect actual per-layer KV transfers from the bundled runtime in memory."""


class GpuReuseTrace:
    def __init__(self):
        self.layers = {}

    def observe(self, event):
        if event.get('event') == 'selective_reuse_h2d_submit':
            self.layers[int(event['layer_id'])] = (
                int(event['keepold_old_page_tokens']),
                int(event['reuse_tokens_h2d']),
            )

    def attach(self, runtime):
        append = runtime._append_metrics

        def collect(event):
            self.observe(event)
            append(event)

        runtime._append_metrics = collect

    def snapshot(self):
        values = list(self.layers.values())
        return dict(
            observed_layers=len(values),
            gpu_reused_tokens_per_layer_min=min((v[0] for v in values), default=0),
            gpu_reused_tokens_per_layer_max=max((v[0] for v in values), default=0),
            cpu_loaded_tokens_per_layer_min=min((v[1] for v in values), default=0),
            cpu_loaded_tokens_per_layer_max=max((v[1] for v in values), default=0),
        )
